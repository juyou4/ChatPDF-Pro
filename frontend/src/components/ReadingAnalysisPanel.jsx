import React, { memo } from 'react';
import { CheckCircle2, Circle, FileText, Languages, Loader2, RefreshCw, StickyNote, Trash2 } from 'lucide-react';

const TYPE_LABEL = {
  heading: '标题',
  paragraph: '段落',
  caption: '图注',
  figure: '图表',
  table: '表格',
};

function ReadingAnalysisPanel({
  blocks = [],
  translations = {},
  translatingBlockIds = [],
  loading = false,
  error = '',
  notice = '',
  pretranslateProgress = { running: false, done: 0, total: 0 },
  onPretranslate,
  currentPage = 1,
  activeBlockId = null,
  notes = [],
  userNotes = [],
  activeNodeId = null,
  visitedNodeIds = [],
  onTranslate,
  onRetranslateBlock,
  onBlockHover,
  onBlockClick,
  onNoteClick,
  onUserNoteClick,
  onDeleteUserNote,
  darkMode = false,
}) {
  const translatableCount = blocks.filter((block) => block?.block_id && block?.text).length;
  const visitedSet = new Set(visitedNodeIds || []);
  const translatingSet = new Set(translatingBlockIds || []);
  const pretranslateTotal = Number(pretranslateProgress?.total || 0);
  const pretranslateDone = Math.min(pretranslateTotal, Number(pretranslateProgress?.done || 0));
  const pretranslatePercent = pretranslateTotal > 0 ? Math.round((pretranslateDone / pretranslateTotal) * 100) : 0;

  return (
    <div className={`h-full flex flex-col ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
      <div className={`px-5 py-4 border-b shrink-0 ${darkMode ? 'border-white/10' : 'border-gray-100'}`}>
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[15px] font-bold tracking-tight">第 {currentPage} 页解析</div>
            <div className={`mt-0.5 text-[11px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              {userNotes.length} 条划词笔记 · {notes.length} 条 AI 要点 · {translatableCount} 个文本块
            </div>
          </div>
          <button
            type="button"
            onClick={onTranslate}
            disabled={loading || translatableCount === 0}
            className={`shrink-0 inline-flex items-center gap-1.5 rounded-full px-4 py-2.5 text-xs font-bold transition-[transform,background-color,box-shadow] duration-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50 disabled:active:scale-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${
              darkMode
                ? 'bg-[#F0653A] text-white shadow-[0_9px_22px_-8px_rgba(240,101,58,0.45)] hover:-translate-y-0.5 hover:bg-[#F5713F] hover:shadow-[0_12px_26px_-8px_rgba(240,101,58,0.52)]'
                : 'accent-cta'
            }`}
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Languages className="w-3.5 h-3.5" />}
            翻译当前页
          </button>
        </div>
        {error && (
          <div className={`mt-3 rounded-lg px-3 py-2 text-xs ${darkMode ? 'bg-red-500/10 text-red-300' : 'bg-red-50 text-red-600'}`}>
            {error}
          </div>
        )}
        {pretranslateTotal > 0 && (
          <div className={`mt-3 rounded-xl border px-3 py-2.5 ${darkMode ? 'border-white/10 bg-white/[0.04]' : 'border-gray-100 bg-white/70'}`}>
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className={`text-[11px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                  悬浮预翻译
                </div>
                <div className={`mt-0.5 text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  {pretranslateProgress.running ? '正在补齐全文缓存' : '已缓存可悬浮翻译'} · {pretranslateDone}/{pretranslateTotal}
                </div>
                {notice && !error && (
                  <div className={`mt-1 text-[10px] font-medium ${darkMode ? 'text-[#fdc4af]' : 'text-[#ed8c68]'}`}>
                    {notice}
                  </div>
                )}
              </div>
              <button
                type="button"
                onClick={() => onPretranslate?.()}
                disabled={pretranslateProgress.running}
                className={`shrink-0 rounded-full px-3.5 py-2 text-[11px] font-bold transition-[transform,background-color,box-shadow] duration-200 hover:-translate-y-0.5 active:translate-y-0 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0 disabled:active:scale-100 focus-visible:outline-none focus-visible:ring-2 ${
                  darkMode
                    ? 'bg-white/10 text-gray-100 shadow-[0_8px_20px_-10px_rgba(0,0,0,0.65)] hover:bg-white/15 focus-visible:ring-white/20'
                    : 'bg-[#111827] text-white shadow-[0_9px_22px_-9px_rgba(17,24,39,0.48)] hover:bg-[#1f2937] hover:shadow-[0_12px_26px_-9px_rgba(17,24,39,0.54)] focus-visible:ring-gray-400/35'
                }`}
              >
                {pretranslateProgress.running ? '处理中' : pretranslateDone > 0 ? '补齐全文' : '开始缓存'}
              </button>
            </div>
            <div className={`mt-2 h-1.5 overflow-hidden rounded-full ${darkMode ? 'bg-white/10' : 'bg-gray-100'}`}>
              <div
                className={`h-full rounded-full transition-all duration-300 ${darkMode ? 'bg-gray-200' : 'bg-gray-900'}`}
                style={{ width: `${pretranslatePercent}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-3">
        {userNotes.length > 0 && (
          <section className="space-y-2.5">
            <div className={`text-[11px] font-bold tracking-wider uppercase ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              我的划词笔记
            </div>
            {userNotes.map((note, index) => (
              <div
                key={note.id}
                className={`group relative overflow-hidden rounded-[10px] border transition-colors ${
                  darkMode
                    ? 'border-amber-300/15 bg-amber-200/[0.06] hover:bg-amber-200/[0.09]'
                    : 'border-amber-200/70 bg-[#fffaf0] hover:border-amber-300 hover:bg-[#fff7e6]'
                }`}
              >
                <button
                  type="button"
                  onClick={() => onUserNoteClick?.(note)}
                  className="w-full px-3 py-3 pr-10 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-amber-300/60"
                >
                  <div className="flex items-start gap-2.5">
                    <StickyNote className={`mt-0.5 h-4 w-4 shrink-0 ${darkMode ? 'text-amber-300/70' : 'text-amber-600'}`} />
                    <div className="min-w-0 flex-1">
                      <div className={`text-[12px] font-semibold leading-relaxed ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>
                        {note.note}
                      </div>
                      <div className={`mt-1.5 line-clamp-2 border-l-2 pl-2 text-[11px] leading-relaxed ${
                        darkMode ? 'border-white/10 text-gray-500' : 'border-amber-200 text-gray-500'
                      }`}>
                        {note.text}
                      </div>
                    </div>
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => onDeleteUserNote?.(note.id)}
                  className={`absolute right-2 top-2 inline-flex h-7 w-7 items-center justify-center rounded-[8px] opacity-0 transition-all group-hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 ${
                    darkMode
                      ? 'text-gray-500 hover:bg-rose-400/10 hover:text-rose-300 focus-visible:ring-white/15'
                      : 'text-gray-400 hover:bg-rose-50 hover:text-rose-600 focus-visible:ring-rose-200'
                  }`}
                  aria-label={`删除第 ${index + 1} 条划词笔记`}
                  title="删除笔记"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </section>
        )}

        {notes.length > 0 && (
          <section className="space-y-2.5">
            <div className={`text-[11px] font-bold tracking-wider uppercase ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              AI 结构化笔记
            </div>
            {notes.map((note) => {
              const isActive = activeNodeId === note.id;
              const isVisited = !isActive && visitedSet.has(note.id);
              return (
                <article
                  key={note.id}
                  onClick={() => onNoteClick?.(note)}
                  className={`rounded-lg border p-3 cursor-pointer transition-all ${
                    isActive
                      ? (darkMode ? 'border-emerald-400/70 bg-emerald-400/10' : 'border-emerald-300 bg-emerald-50')
                      : isVisited
                        ? (darkMode ? 'border-amber-400/50 bg-amber-400/10' : 'border-amber-200 bg-amber-50/80')
                        : (darkMode ? 'border-white/10 bg-white/[0.04] hover:bg-white/[0.07]' : 'border-gray-100 bg-white/80 hover:border-gray-200 hover:bg-white')
                  }`}
                >
                  <div className="flex items-start gap-2">
                    {isActive ? (
                      <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
                    ) : isVisited ? (
                      <Circle className="mt-0.5 h-4 w-4 shrink-0 fill-amber-400 text-amber-400" />
                    ) : (
                      <FileText className={`mt-0.5 h-4 w-4 shrink-0 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`} />
                    )}
                    <div className="min-w-0 flex-1">
                      <div className={`text-[13px] font-bold leading-snug ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                        {note.title}
                      </div>
                      {note.summary && (
                        <div className={`mt-1 text-[12px] leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          {note.summary}
                        </div>
                      )}
                      <div className={`mt-2 text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                        证据 {note.evidence_block_ids?.length || note.evidence?.block_ids?.length || 0} 处
                      </div>
                    </div>
                  </div>
                </article>
              );
            })}
          </section>
        )}

        {blocks.length === 0 ? (
          notes.length === 0 && userNotes.length === 0 ? (
          <div className={`h-full flex flex-col items-center justify-center text-center ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            <FileText className="w-8 h-8 mb-3 opacity-60" />
            <div className="text-sm font-medium">暂无可解析段落</div>
          </div>
          ) : null
        ) : (
          <>
          <div className={`pt-2 text-[11px] font-bold tracking-wider uppercase ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            当前页原文
          </div>
          {blocks.map((block) => {
            const item = translations[block.block_id];
            const isActive = activeBlockId === block.block_id;
            const isTranslatingBlock = translatingSet.has(block.block_id);
            return (
              <article
                key={block.block_id}
                onMouseEnter={() => onBlockHover?.(block)}
                onMouseLeave={() => onBlockHover?.(null)}
                onClick={() => onBlockClick?.(block)}
                className={`rounded-lg border p-3 cursor-pointer transition-all ${
                  isActive
                    ? (darkMode ? 'border-amber-400/70 bg-amber-400/10' : 'border-amber-300 bg-amber-50')
                    : (darkMode ? 'border-white/10 bg-white/[0.04] hover:bg-white/[0.07]' : 'border-gray-100 bg-white/80 hover:border-gray-200 hover:bg-white')
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div className="min-w-0 flex items-center gap-2">
                    <span className={`rounded-md px-1.5 py-0.5 text-[10px] font-bold ${darkMode ? 'bg-white/10 text-gray-300' : 'bg-gray-100 text-gray-500'}`}>
                      {TYPE_LABEL[block.type] || '文本'}
                    </span>
                    <span className={`truncate text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      {block.block_id}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      onRetranslateBlock?.(block);
                    }}
                    disabled={isTranslatingBlock}
                    className={`shrink-0 inline-flex items-center gap-1 rounded-md px-2 py-1 text-[10px] font-bold transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                      darkMode
                        ? 'bg-white/10 text-gray-200 hover:bg-white/15'
                        : 'bg-white text-gray-500 shadow-sm ring-1 ring-gray-100 hover:text-[#ed8c68] hover:ring-[#feded2]'
                    }`}
                  >
                    {isTranslatingBlock ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <RefreshCw className="h-3 w-3" />
                    )}
                    {isTranslatingBlock ? '翻译中' : item ? '重译' : '翻译本段'}
                  </button>
                </div>
                <div className={`text-[12px] leading-relaxed line-clamp-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {block.text}
                </div>
                {item ? (
                  <div className={`mt-3 border-t pt-3 ${darkMode ? 'border-white/10' : 'border-gray-100'}`}>
                    {item.summary && (
                      <div className={`mb-2 text-[12px] font-semibold leading-relaxed ${darkMode ? 'text-amber-200' : 'text-amber-700'}`}>
                        {item.summary}
                      </div>
                    )}
                    <div className={`text-[13px] leading-relaxed ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>
                      {item.translation}
                    </div>
                  </div>
                ) : (
                  <div className={`mt-3 text-[11px] font-medium ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                    未翻译
                  </div>
                )}
              </article>
            );
          })}
          </>
        )}
      </div>
    </div>
  );
}

export default memo(ReadingAnalysisPanel);
