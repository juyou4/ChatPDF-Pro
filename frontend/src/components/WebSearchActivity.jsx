import React, { memo, useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, ExternalLink, Globe2, Loader2, Search } from 'lucide-react';

const COLLAPSED_SOURCE_COUNT = 3;

const normalizeText = (value, maxLength = 240) => String(value || '')
  .replace(/\s+/g, ' ')
  .trim()
  .slice(0, maxLength);

const getSafeHttpUrl = (value) => {
  const raw = String(value || '').trim();
  if (!raw) return '';
  try {
    const parsed = new URL(raw);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.toString() : '';
  } catch {
    return '';
  }
};

const getSourceDomain = (url, fallback = '') => {
  try {
    return new URL(url).hostname.replace(/^www\./i, '');
  } catch {
    return normalizeText(fallback, 80);
  }
};

const getAdapterLabel = (adapter) => {
  const normalized = String(adapter || '').trim().toLowerCase();
  if (normalized === 'github_public') return 'GitHub';
  if (normalized === 'youtube_transcript') return 'YouTube';
  return '';
};

const getAuditMessage = (audit) => {
  const status = String(audit?.status || '').trim().toLowerCase();
  if (status === 'empty') return '没有找到足够可靠的网页来源，回答将仅使用现有文档证据。';
  if (status === 'failed') return '网页搜索未完成，回答已回退到当前可用证据。';
  if (audit?.reason === 'missing_topic') return '还需要更明确的主题才能执行网页搜索。';
  if (audit?.reason === 'auto_policy_not_selected') return '当前问题未触发外部搜索，本轮仅使用文档与对话上下文。';
  if (audit?.reason === 'agent_not_selected') return '模型判断本轮无需外部搜索，本轮仅使用文档与对话上下文。';
  if (status === 'skipped') return '本轮未执行网页搜索。';
  return '';
};

const WebSearchActivity = ({
  sources = [],
  reads = [],
  status = null,
  audit = null,
  query = '',
  isStreaming = false,
  darkMode = false,
  embedded = false,
}) => {
  const normalizedSources = useMemo(() => (
    (Array.isArray(sources) ? sources : [])
      .map((source, index) => {
        const safeUrl = getSafeHttpUrl(source?.url);
        const domain = getSourceDomain(safeUrl, source?.domain || source?.source);
        const title = normalizeText(source?.title || domain || `来源 ${index + 1}`, 180);
        return {
          id: `${safeUrl || domain || title}-${index}`,
          sourceId: String(source?.source_id || source?.evidence_id || '').trim(),
          url: safeUrl,
          title,
          domain,
          adapter: String(source?.adapter || '').trim(),
        };
      })
      .filter((source) => source.title)
  ), [sources]);

  const normalizedReads = useMemo(() => (
    (Array.isArray(reads) ? reads : [])
      .map((read) => ({
        sourceId: String(read?.source_id || '').trim(),
        evidenceId: String(read?.evidence_id || '').trim(),
        url: getSafeHttpUrl(read?.url),
        status: String(read?.status || '').trim().toLowerCase(),
        charCount: Number.isFinite(Number(read?.char_count)) ? Number(read.char_count) : 0,
        truncated: Boolean(read?.truncated),
        cached: Boolean(read?.cached),
        adapter: String(read?.adapter || '').trim(),
        contentKind: String(read?.content_kind || '').trim(),
      }))
      .filter((read) => read.sourceId || read.evidenceId)
  ), [reads]);
  const readBySource = useMemo(() => {
    const map = new Map();
    normalizedReads.forEach((read) => {
      if (read.sourceId) map.set(`id:${read.sourceId}`, read);
      if (read.url) map.set(`url:${read.url}`, read);
    });
    return map;
  }, [normalizedReads]);

  const phase = String(status?.phase || '').trim().toLowerCase();
  const auditStatus = String(audit?.status || '').trim().toLowerCase();
  const isSearching = Boolean(isStreaming && phase === 'searching');
  const sourceCount = Math.max(
    normalizedSources.length,
    Number.isFinite(Number(status?.count)) ? Number(status.count) : 0,
    Number.isFinite(Number(audit?.result_count)) ? Number(audit.result_count) : 0,
  );
  const completed = normalizedSources.length > 0 || normalizedReads.length > 0 || auditStatus === 'completed';
  const requested = Boolean(status || normalizedSources.length > 0 || normalizedReads.length > 0 || audit?.requested);
  const auditMessage = getAuditMessage(audit);
  const [expanded, setExpanded] = useState(() => (
    isSearching || Boolean(auditMessage) || (!embedded && normalizedSources.length > 0)
  ));
  const [showAllSources, setShowAllSources] = useState(false);
  const wasSearchingRef = useRef(isSearching);

  useEffect(() => {
    if (isSearching || auditMessage || (!embedded && normalizedSources.length > 0)) setExpanded(true);
    if (wasSearchingRef.current && !isSearching) setExpanded(true);
    wasSearchingRef.current = isSearching;
  }, [auditMessage, embedded, isSearching, normalizedSources.length]);

  useEffect(() => {
    setShowAllSources(false);
  }, [normalizedSources.length]);

  if (!requested) return null;

  const title = isSearching
    ? '正在搜索网页'
    : completed
      ? `已检索 ${sourceCount || normalizedSources.length} 个网页`
      : auditStatus === 'empty'
        ? '未找到可用网页来源'
        : auditStatus === 'failed'
          ? '网页搜索未完成'
          : '网页搜索';
  const visibleSources = showAllSources
    ? normalizedSources
    : normalizedSources.slice(0, COLLAPSED_SOURCE_COUNT);
  const hiddenSourceCount = Math.max(0, normalizedSources.length - visibleSources.length);
  const displayQuery = normalizeText(query, 240);
  const successfulReadCount = normalizedReads.filter((read) => read.status === 'completed').length;
  const failedReadCount = normalizedReads.filter((read) => read.status && read.status !== 'completed').length;
  const readLabel = (source) => {
    const read = source?.sourceId
      ? readBySource.get(`id:${source.sourceId}`)
      : readBySource.get(`url:${source?.url || ''}`);
    if (!read) return '';
    if (read.status === 'completed') {
      if (read.contentKind === 'youtube_metadata') return '已读取视频信息';
      if (read.contentKind === 'rendered_web_page') return '已读取动态页面';
      return read.charCount > 0 ? `已读取 ${read.charCount.toLocaleString()} 字` : '已读取全文';
    }
    return '全文读取失败，使用搜索摘要';
  };

  if (embedded) {
    return (
      <div className="agent-op-enter relative min-h-10 py-0.5 pl-1" data-testid="web-search-activity">
        <span
          className={`absolute -left-[31px] top-[9px] z-[1] grid h-[22px] w-[22px] place-items-center rounded-full border ring-[3px] ${
            isSearching
              ? darkMode
                ? 'border-[#FFA07A] bg-[#FFA07A] text-[#24272d] ring-[#2b2e34]'
                : 'border-[#a8624e] bg-[#a8624e] text-white ring-[#faf8f6]'
              : darkMode
                ? 'border-white/15 bg-[#444850] text-gray-100 ring-[#2b2e34]'
                : 'border-[#d8cec7] bg-[#f2ece7] text-[#5c5049] ring-[#faf8f6]'
          } ${isSearching ? 'agent-timeline-node-active' : ''}`}
          aria-hidden="true"
        >
          {isSearching ? (
            <Loader2 className="h-[15px] w-[15px] animate-spin motion-reduce:animate-none" strokeWidth={2.35} />
          ) : completed ? (
            <Globe2 className="h-[15px] w-[15px]" strokeWidth={2.15} />
          ) : (
            <Search className="h-[15px] w-[15px]" strokeWidth={2.15} />
          )}
        </span>

        <button
          type="button"
          onClick={() => setExpanded((current) => !current)}
          aria-expanded={expanded}
          className={`flex min-h-9 w-full items-center gap-2.5 rounded-[8px] px-2.5 py-1.5 text-left transition-[background-color,transform] duration-200 active:scale-[0.995] focus-visible:outline-none focus-visible:ring-2 ${
            darkMode
              ? 'hover:bg-white/[0.035] focus-visible:ring-[#FFA07A]/35'
              : 'hover:bg-[#f6f3f1] focus-visible:ring-[#D99178]/35'
          }`}
        >
          <span className={`min-w-0 flex-1 truncate text-[13.5px] ${
            isSearching
              ? darkMode ? 'font-medium text-gray-200' : 'font-medium text-[#4f4a46]'
              : darkMode ? 'text-gray-200' : 'text-[#514a45]'
          }`}>
            {title}
          </span>
          <ChevronDown
            className={`h-4 w-4 flex-shrink-0 transition-transform duration-300 ${
              darkMode ? 'text-gray-300' : 'text-[#7c7069]'
            } ${expanded ? 'rotate-180' : ''}`}
            strokeWidth={1.9}
            aria-hidden="true"
          />
        </button>

        <div
          className={`grid transition-[grid-template-rows,opacity] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none ${
            expanded ? 'grid-rows-[1fr] opacity-100' : 'pointer-events-none grid-rows-[0fr] opacity-0'
          }`}
          aria-hidden={!expanded}
        >
          <div className="min-h-0 overflow-hidden">
            <div className="pb-1 pl-1 pr-1 pt-0.5">
              {displayQuery && (
                <div className={`flex items-start gap-2.5 rounded-[7px] px-2 py-2 text-[12.5px] leading-5 ${darkMode ? 'text-gray-300' : 'text-[#67615d]'}`}>
                  <Search className="mt-[2px] h-3.5 w-3.5 flex-shrink-0" strokeWidth={1.8} aria-hidden="true" />
                  <span className="line-clamp-2" title={displayQuery}>已搜索网络：{displayQuery}</span>
                </div>
              )}
              {visibleSources.map((source) => {
                const sourceContent = (
                  <>
                    <Globe2 className={`h-3.5 w-3.5 flex-shrink-0 ${darkMode ? 'text-gray-400' : 'text-[#918983]'}`} strokeWidth={1.8} aria-hidden="true" />
                    <span className={`min-w-0 truncate ${darkMode ? 'text-gray-300' : 'text-[#67615d]'}`}>{source.title}</span>
                    {getAdapterLabel(source.adapter) && <span className={`flex-shrink-0 text-[10.5px] ${darkMode ? 'text-gray-500' : 'text-[#9a8f88]'}`}>{getAdapterLabel(source.adapter)}</span>}
                    {source.domain && <span className={`min-w-0 flex-shrink truncate text-[11.5px] ${darkMode ? 'text-gray-500' : 'text-[#918983]'}`}>{source.domain}</span>}
                    {readLabel(source) && <span className={`ml-auto min-w-0 flex-shrink truncate text-[11px] ${readLabel(source).startsWith('已读取') ? (darkMode ? 'text-emerald-300/70' : 'text-emerald-700/75') : (darkMode ? 'text-amber-300/75' : 'text-amber-700/80')}`}>{readLabel(source)}</span>}
                    {source.url && <ExternalLink className="ml-auto h-3.5 w-3.5 flex-shrink-0 opacity-0 transition-opacity group-hover/source:opacity-100" strokeWidth={1.8} aria-hidden="true" />}
                  </>
                );
                const sourceClass = `group/source flex min-h-8 w-full items-center gap-2.5 rounded-[7px] px-2 py-1.5 text-left text-[12.5px] transition-colors duration-200 ${
                  darkMode ? 'hover:bg-white/[0.035]' : 'hover:bg-[#f6f3f1]'
                }`;
                return source.url ? (
                  <a key={source.id} href={source.url} target="_blank" rel="noopener noreferrer" className={sourceClass}>{sourceContent}</a>
                ) : (
                  <div key={source.id} className={sourceClass}>{sourceContent}</div>
                );
              })}
              {hiddenSourceCount > 0 && (
                <button
                  type="button"
                  onClick={() => setShowAllSources(true)}
                  className={`rounded-[7px] px-2 py-1.5 text-[12px] transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                    darkMode
                      ? 'text-gray-500 hover:bg-white/[0.04] hover:text-gray-300 focus-visible:ring-[#FFA07A]/35'
                      : 'text-gray-400 hover:bg-[#f6f3f1] hover:text-gray-600 focus-visible:ring-[#D99178]/35'
                  }`}
                >
                  另外 {hiddenSourceCount} 个来源
                </button>
              )}
              {normalizedReads.length > 0 && (
                <p className={`px-2 py-1 text-[11.5px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  {successfulReadCount > 0 ? `已读取 ${successfulReadCount} 个来源全文${failedReadCount > 0 ? `，${failedReadCount} 个来源读取失败` : ''}` : '未能读取来源全文，已使用搜索摘要'}
                </p>
              )}
              {!isSearching && normalizedSources.length === 0 && auditMessage && (
                <p className={`px-2 py-1.5 text-[12.5px] leading-5 ${
                  auditStatus === 'failed' || auditStatus === 'empty'
                    ? darkMode ? 'text-amber-300/80' : 'text-amber-700'
                    : darkMode ? 'text-gray-500' : 'text-gray-500'
                }`}>
                  {auditMessage}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <section
      className={`mb-4 mt-1 w-full max-w-[46rem] text-[13px] ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}
      data-testid="web-search-activity"
    >
      <button
        type="button"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
        className={`group -ml-1 flex min-h-8 max-w-full items-center gap-2 rounded-[9px] px-1.5 py-1 text-left transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 ${
          darkMode
            ? 'hover:bg-white/[0.04] focus-visible:ring-[#FFA07A]/40'
            : 'hover:bg-[#f6f3f1] focus-visible:ring-[#D99178]/40'
        }`}
      >
        <span className={`grid h-5 w-5 flex-shrink-0 place-items-center ${darkMode ? 'text-gray-400' : 'text-[#8b817b]'}`} aria-hidden="true">
          {isSearching ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin motion-reduce:animate-none" strokeWidth={1.9} />
          ) : completed ? (
            <Check className="h-3.5 w-3.5" strokeWidth={2.2} />
          ) : (
            <Globe2 className="h-3.5 w-3.5" strokeWidth={1.8} />
          )}
        </span>
        <span className={`truncate text-[12.5px] font-medium ${darkMode ? 'text-gray-300' : 'text-[#5f5a56]'}`}>
          {title}
        </span>
        {isSearching && (
          <span className={`h-1.5 w-1.5 flex-shrink-0 animate-pulse rounded-full motion-reduce:animate-none ${darkMode ? 'bg-[#FFA07A]' : 'bg-[#B85F47]'}`} aria-hidden="true" />
        )}
        <ChevronDown
          className={`h-3.5 w-3.5 flex-shrink-0 transition-transform duration-300 ${
            darkMode ? 'text-gray-500' : 'text-gray-400'
          } ${expanded ? 'rotate-180' : ''}`}
          strokeWidth={1.9}
          aria-hidden="true"
        />
      </button>

      <div
        className={`grid transition-[grid-template-rows,opacity,visibility] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none ${
          expanded
            ? 'visible grid-rows-[1fr] opacity-100'
            : 'invisible pointer-events-none grid-rows-[0fr] opacity-0'
        }`}
        aria-hidden={!expanded}
        inert={expanded ? undefined : ''}
      >
        <div className="min-h-0 overflow-hidden">
          <div className={`relative ml-[9px] border-l pb-1 pl-5 pt-1.5 ${darkMode ? 'border-white/[0.10]' : 'border-[#ddd9d6]'}`}>
            {displayQuery && (
              <div className={`mb-1.5 flex min-h-7 items-start gap-2 rounded-[7px] px-1.5 py-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <Search className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" strokeWidth={1.8} aria-hidden="true" />
                <span className="line-clamp-2 leading-5" title={displayQuery}>{displayQuery}</span>
              </div>
            )}

            <div className="space-y-0.5">
              {visibleSources.map((source) => {
                const content = (
                  <>
                    <span className={`grid h-4 w-4 flex-shrink-0 place-items-center rounded-[5px] border ${
                      darkMode
                        ? 'border-white/[0.10] bg-white/[0.045] text-gray-400'
                        : 'border-[#e3dfdc] bg-[#f8f6f4] text-[#8d847e]'
                    }`} aria-hidden="true">
                      <Globe2 className="h-2.5 w-2.5" strokeWidth={1.9} />
                    </span>
                    <span className={`min-w-0 truncate font-medium ${darkMode ? 'text-gray-300' : 'text-[#4f4b48]'}`}>
                      {source.title}
                    </span>
                    {getAdapterLabel(source.adapter) && (
                      <span className={`flex-shrink-0 text-[10.5px] ${darkMode ? 'text-gray-500' : 'text-[#9a8f88]'}`}>
                        {getAdapterLabel(source.adapter)}
                      </span>
                    )}
                    {source.domain && (
                      <span className={`min-w-0 flex-shrink truncate text-[11.5px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                        {source.domain}
                      </span>
                    )}
                    {readLabel(source) && <span className={`ml-auto min-w-0 flex-shrink truncate text-[11px] ${readLabel(source).startsWith('已读取') ? (darkMode ? 'text-emerald-300/70' : 'text-emerald-700/75') : (darkMode ? 'text-amber-300/75' : 'text-amber-700/80')}`}>{readLabel(source)}</span>}
                    {source.url && (
                      <ExternalLink className="ml-auto h-3 w-3 flex-shrink-0 opacity-0 transition-opacity group-hover/source:opacity-100" strokeWidth={1.8} aria-hidden="true" />
                    )}
                  </>
                );
                const rowClass = `group/source flex min-h-7 w-full items-center gap-2 rounded-[7px] px-1.5 py-1 text-left transition-colors duration-200 ${
                  darkMode ? 'hover:bg-white/[0.04]' : 'hover:bg-[#f6f3f1]'
                }`;
                return source.url ? (
                  <a key={source.id} href={source.url} target="_blank" rel="noopener noreferrer" className={rowClass}>
                    {content}
                  </a>
                ) : (
                  <div key={source.id} className={rowClass}>{content}</div>
                );
              })}
            </div>

            {hiddenSourceCount > 0 && (
              <button
                type="button"
                onClick={() => setShowAllSources(true)}
                className={`mt-0.5 rounded-[7px] px-1.5 py-1 text-[11.5px] transition-colors focus-visible:outline-none focus-visible:ring-2 ${
                  darkMode
                    ? 'text-gray-500 hover:bg-white/[0.04] hover:text-gray-300 focus-visible:ring-[#FFA07A]/35'
                    : 'text-gray-400 hover:bg-[#f6f3f1] hover:text-gray-600 focus-visible:ring-[#D99178]/35'
                }`}
              >
                另外 {hiddenSourceCount} 个来源
              </button>
            )}

            {normalizedReads.length > 0 && (
              <p className={`px-1.5 py-1 text-[11.5px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                {successfulReadCount > 0 ? `已读取 ${successfulReadCount} 个来源全文${failedReadCount > 0 ? `，${failedReadCount} 个来源读取失败` : ''}` : '未能读取来源全文，已使用搜索摘要'}
              </p>
            )}

            {!isSearching && normalizedSources.length === 0 && auditMessage && (
              <p className={`px-1.5 py-1 text-[12px] leading-5 ${
                auditStatus === 'failed' || auditStatus === 'empty'
                  ? darkMode ? 'text-amber-300/80' : 'text-amber-700'
                  : darkMode ? 'text-gray-500' : 'text-gray-500'
              }`}>
                {auditMessage}
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default memo(WebSearchActivity);
