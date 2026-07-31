import React, { memo, useCallback, useEffect, useRef } from 'react';
import { Compartment, EditorState } from '@codemirror/state';
import { EditorView, keymap, placeholder as cmPlaceholder } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { markdown, markdownLanguage } from '@codemirror/lang-markdown';
import {
  Bold,
  Braces,
  CheckSquare,
  Code2,
  Heading2,
  Italic,
  Link2,
  List,
  Quote,
} from 'lucide-react';
import { markdownLivePreview, markdownNoteTheme } from '../utils/markdownLivePreview';

// 常用的排在前面；写笔记时高频的是标题、列表、待办，粗斜体次之。
const FORMAT_ACTIONS = [
  { id: 'heading', label: '二级标题', icon: Heading2 },
  { id: 'list', label: '无序列表', icon: List },
  { id: 'task', label: '任务清单', icon: CheckSquare },
  { id: 'bold', label: '粗体', icon: Bold },
  { id: 'italic', label: '斜体', icon: Italic },
  { id: 'quote', label: '引用', icon: Quote },
  { id: 'link', label: '链接', icon: Link2 },
  { id: 'code', label: '代码块', icon: Code2 },
  { id: 'math', label: '行内公式', icon: Braces },
];

const PLACEHOLDER = '记录问题、结论或待办…';

const wrapSelection = (value, start, end, before, after, placeholder) => {
  const selected = value.slice(start, end) || placeholder;
  return {
    value: `${value.slice(0, start)}${before}${selected}${after}${value.slice(end)}`,
    start: start + before.length,
    end: start + before.length + selected.length,
  };
};

const prefixSelectedLines = (value, start, end, prefix) => {
  const blockStart = start > 0 ? value.lastIndexOf('\n', start - 1) + 1 : 0;
  const nextLine = value.indexOf('\n', end);
  const blockEnd = nextLine === -1 ? value.length : nextLine;
  const selectedBlock = value.slice(blockStart, blockEnd) || '';
  const nextBlock = selectedBlock
    .split('\n')
    .map((line) => `${prefix}${line}`)
    .join('\n');
  return {
    value: `${value.slice(0, blockStart)}${nextBlock}${value.slice(blockEnd)}`,
    start: blockStart + prefix.length,
    end: blockStart + nextBlock.length,
  };
};

export const applyMarkdownNoteFormat = (value, start, end, action) => {
  const source = String(value || '');
  const selectionStart = Math.max(0, Number(start) || 0);
  const selectionEnd = Math.max(selectionStart, Number(end) || selectionStart);

  switch (action) {
    case 'bold':
      return wrapSelection(source, selectionStart, selectionEnd, '**', '**', '重点');
    case 'italic':
      return wrapSelection(source, selectionStart, selectionEnd, '*', '*', '文字');
    case 'heading':
      return prefixSelectedLines(source, selectionStart, selectionEnd, '## ');
    case 'list':
      return prefixSelectedLines(source, selectionStart, selectionEnd, '- ');
    case 'task':
      return prefixSelectedLines(source, selectionStart, selectionEnd, '- [ ] ');
    case 'quote':
      return prefixSelectedLines(source, selectionStart, selectionEnd, '> ');
    case 'code':
      return wrapSelection(source, selectionStart, selectionEnd, '```\n', '\n```', '代码');
    case 'link': {
      const selected = source.slice(selectionStart, selectionEnd) || '链接文字';
      const replacement = `[${selected}](https://)`;
      return {
        value: `${source.slice(0, selectionStart)}${replacement}${source.slice(selectionEnd)}`,
        start: selectionStart + selected.length + 3,
        end: selectionStart + selected.length + 11,
      };
    }
    case 'math':
      return wrapSelection(source, selectionStart, selectionEnd, '$', '$', 'x');
    default:
      return { value: source, start: selectionStart, end: selectionEnd };
  }
};

function MarkdownNoteEditor({
  value,
  onChange,
  darkMode = false,
  maxLength = 20000,
  autoFocus = false,
  fill = false,
  editorLabel = 'Markdown 笔记编辑器',
  onSubmit,
  onCancel,
}) {
  const hostRef = useRef(null);
  const viewRef = useRef(null);
  const themeRef = useRef(new Compartment());
  const source = String(value || '');

  // 键盘和变更回调走 ref，编辑器实例才能只创建一次。
  const handlersRef = useRef({ onChange, onSubmit, onCancel });
  handlersRef.current = { onChange, onSubmit, onCancel };

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;

    const view = new EditorView({
      parent: host,
      state: EditorState.create({
        doc: String(value || ''),
        extensions: [
          history(),
          keymap.of([
            {
              key: 'Mod-Enter',
              run: () => {
                if (typeof handlersRef.current.onSubmit !== 'function') return false;
                handlersRef.current.onSubmit();
                return true;
              },
            },
            {
              key: 'Escape',
              run: () => {
                if (typeof handlersRef.current.onCancel !== 'function') return false;
                handlersRef.current.onCancel();
                return true;
              },
            },
            ...historyKeymap,
            ...defaultKeymap,
          ]),
          markdown({ base: markdownLanguage }),
          markdownLivePreview(),
          EditorView.lineWrapping,
          cmPlaceholder(PLACEHOLDER),
          EditorState.changeFilter.of((transaction) => transaction.newDoc.length <= maxLength),
          EditorView.contentAttributes.of({ 'aria-label': editorLabel }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) handlersRef.current.onChange?.(update.state.doc.toString());
          }),
          themeRef.current.of(markdownNoteTheme(darkMode)),
        ],
      }),
    });

    viewRef.current = view;
    if (autoFocus) view.focus();
    return () => {
      view.destroy();
      viewRef.current = null;
    };
    // 只挂载一次：文档、主题都通过 dispatch / compartment 更新
  }, []);

  // 外部改写 value（切换文档、保存后清空）时同步回编辑器
  useEffect(() => {
    const view = viewRef.current;
    if (!view || view.state.doc.toString() === source) return;
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: source } });
  }, [source]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({ effects: themeRef.current.reconfigure(markdownNoteTheme(darkMode)) });
  }, [darkMode]);

  const handleFormat = useCallback((action) => {
    const view = viewRef.current;
    if (!view) return;
    const current = view.state.doc.toString();
    const { from, to } = view.state.selection.main;
    const formatted = applyMarkdownNoteFormat(current, from, to, action);
    if (formatted.value.length > maxLength) return;
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: formatted.value },
      selection: { anchor: formatted.start, head: formatted.end },
    });
    view.focus();
  }, [maxLength]);

  const remaining = maxLength - source.length;
  // 字数只在快满时才提示，平时不占视线。
  const showCounter = remaining <= Math.max(200, Math.round(maxLength * 0.05));

  return (
    <div className={`markdown-note-editor flex min-w-0 flex-col ${fill ? 'min-h-0 flex-1' : ''}`}>
      <div
        className={`note-writing-toolbar flex shrink-0 flex-wrap items-center gap-0.5 border-b pb-1.5 ${darkMode ? 'border-white/[0.07] text-gray-400' : 'border-[#ece6e0] text-[#82766a]'}`}
        role="toolbar"
        aria-label="Markdown 格式工具栏"
      >
        {FORMAT_ACTIONS.map((action) => {
          const Icon = action.icon;
          return (
            <button
              key={action.id}
              type="button"
              onClick={() => handleFormat(action.id)}
              className={`inline-flex h-7 w-7 items-center justify-center rounded-[9px] transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 ${
                darkMode
                  ? 'hover:bg-white/[0.08] hover:text-gray-100 focus-visible:ring-white/20'
                  : 'hover:bg-[#f1ece7] hover:text-[#5c534b] focus-visible:ring-[#D97A5D]/25'
              }`}
              aria-label={action.label}
              title={action.label}
            >
              <Icon className="h-3.5 w-3.5" strokeWidth={1.9} />
            </button>
          );
        })}
      </div>

      <div
        ref={hostRef}
        data-testid="markdown-note-editor"
        className={`markdown-note-surface custom-scrollbar min-w-0 overflow-hidden ${
          fill ? 'markdown-note-surface--fill min-h-0 flex-1' : 'min-h-[120px] max-h-[300px]'
        }`}
      />

      {showCounter && (
        <div className={`mt-2 shrink-0 text-right text-[10px] tabular-nums ${
          remaining <= 0 ? 'text-rose-500' : (darkMode ? 'text-gray-600' : 'text-[#8f8479]')
        }`}>
          还可输入 {Math.max(0, remaining)} 字
        </div>
      )}
    </div>
  );
}

export default memo(MarkdownNoteEditor);
