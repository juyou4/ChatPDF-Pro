import { HighlightStyle, syntaxHighlighting, syntaxTree } from '@codemirror/language';
import { Decoration, EditorView, ViewPlugin } from '@codemirror/view';
import { tags } from '@lezer/highlight';

// 就地渲染的核心：语法标记（** # > 等）在光标不在该行时隐藏，
// 光标落到某行就把该行还原成原始 Markdown，方便继续编辑 —— 与 Obsidian 实时预览一致。

const HIDDEN_MARK = Decoration.replace({});

const HEADING_LINE = {
  ATXHeading1: Decoration.line({ class: 'cm-note-line-h1' }),
  ATXHeading2: Decoration.line({ class: 'cm-note-line-h2' }),
  ATXHeading3: Decoration.line({ class: 'cm-note-line-h3' }),
  ATXHeading4: Decoration.line({ class: 'cm-note-line-h4' }),
  ATXHeading5: Decoration.line({ class: 'cm-note-line-h5' }),
  ATXHeading6: Decoration.line({ class: 'cm-note-line-h6' }),
};

const QUOTE_LINE = Decoration.line({ class: 'cm-note-line-quote' });

const CONTENT_MARK = {
  StrongEmphasis: Decoration.mark({ class: 'cm-note-strong' }),
  Emphasis: Decoration.mark({ class: 'cm-note-em' }),
  Strikethrough: Decoration.mark({ class: 'cm-note-strike' }),
  InlineCode: Decoration.mark({ class: 'cm-note-code' }),
  Link: Decoration.mark({ class: 'cm-note-link' }),
};

// ListMark 不隐藏——它就是可见的项目符号；隐藏它会让列表看起来像普通段落。
const HIDEABLE_MARKS = new Set([
  'HeaderMark',
  'EmphasisMark',
  'StrongEmphasisMark',
  'StrikethroughMark',
  'LinkMark',
  'QuoteMark',
]);

const STYLED_MARKS = {
  ListMark: 'cm-note-bullet',
  TaskMarker: 'cm-note-task',
  URL: 'cm-note-url',
  CodeInfo: 'cm-note-code-info',
};

function collectActiveLines(state) {
  const active = new Set();
  state.selection.ranges.forEach((range) => {
    const first = state.doc.lineAt(range.from).number;
    const last = state.doc.lineAt(range.to).number;
    for (let line = first; line <= last; line += 1) active.add(line);
  });
  return active;
}

// 纯函数，只依赖 EditorState + 可见区间，便于无 DOM 直接测试。
export function computeNoteDecorations(state, visibleRanges) {
  const activeLines = collectActiveLines(state);
  const ranges = [];
  const seenLines = new Set();

  const addLineDecoration = (pos, decoration) => {
    const line = state.doc.lineAt(pos);
    const key = `${decoration.spec.class}:${line.number}`;
    if (seenLines.has(key)) return;
    seenLines.add(key);
    ranges.push(decoration.range(line.from));
  };

  visibleRanges.forEach(({ from, to }) => {
    syntaxTree(state).iterate({
      from,
      to,
      enter: (node) => {
        const name = node.name;

        if (HEADING_LINE[name]) {
          addLineDecoration(node.from, HEADING_LINE[name]);
          return;
        }
        if (name === 'Blockquote') {
          // 引用可以跨多行，逐行加左边条
          const first = state.doc.lineAt(node.from).number;
          const last = state.doc.lineAt(node.to).number;
          for (let n = first; n <= last; n += 1) {
            addLineDecoration(state.doc.line(n).from, QUOTE_LINE);
          }
          return;
        }
        if (CONTENT_MARK[name] && node.to > node.from) {
          ranges.push(CONTENT_MARK[name].range(node.from, node.to));
          return;
        }
        if (STYLED_MARKS[name] && node.to > node.from) {
          ranges.push(Decoration.mark({ class: STYLED_MARKS[name] }).range(node.from, node.to));
          return;
        }

        if (!HIDEABLE_MARKS.has(name) && name !== 'CodeMark') return;
        // 围栏代码块的 ``` 保留，否则看不出代码块边界；只折叠行内代码的反引号。
        if (name === 'CodeMark' && node.node.parent?.name !== 'InlineCode') return;
        if (node.to <= node.from) return;
        if (activeLines.has(state.doc.lineAt(node.from).number)) return;

        ranges.push(HIDDEN_MARK.range(node.from, node.to));
      },
    });
  });

  return Decoration.set(ranges, true);
}

export const markdownLivePreview = () => ViewPlugin.fromClass(
  class {
    constructor(view) {
      this.decorations = computeNoteDecorations(view.state, view.visibleRanges);
    }

    update(update) {
      if (update.docChanged || update.viewportChanged || update.selectionSet) {
        this.decorations = computeNoteDecorations(update.view.state, update.view.visibleRanges);
      }
    }
  },
  { decorations: (plugin) => plugin.decorations }
);

const noteHighlightStyle = HighlightStyle.define([
  { tag: tags.monospace, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' },
  { tag: tags.keyword, color: '#b85f47' },
  { tag: tags.string, color: '#6b7f4f' },
  { tag: tags.comment, color: '#82766a', fontStyle: 'italic' },
]);

export const markdownNoteTheme = (darkMode) => [
  syntaxHighlighting(noteHighlightStyle),
  EditorView.theme({
    '&': {
      fontSize: '12.5px',
      backgroundColor: 'transparent',
      color: darkMode ? '#e5e7eb' : '#3a332e',
    },
    '&.cm-focused': { outline: 'none' },
    '.cm-scroller': {
      fontFamily: 'inherit',
      lineHeight: '1.82',
      padding: '0',
      overflowY: 'auto',
      overflowX: 'hidden',
    },
    '.cm-content': { minHeight: '8.5rem', padding: '0.55rem 0 1.25rem', caretColor: '#F0653A' },
    '.cm-line': { padding: '0' },
    '.cm-activeLine': { backgroundColor: 'transparent' },
    '.cm-cursor, .cm-dropCursor': { borderLeftColor: '#F0653A', borderLeftWidth: '2px' },
    '.cm-placeholder': { color: darkMode ? '#6b7280' : '#8f8479' },
    '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': {
      backgroundColor: darkMode ? 'rgba(240, 101, 58, 0.24)' : 'rgba(240, 101, 58, 0.16)',
    },

    '.cm-note-line-h1': { fontSize: '17px', fontWeight: '600', lineHeight: '1.5', paddingTop: '4px' },
    '.cm-note-line-h2': { fontSize: '15px', fontWeight: '600', lineHeight: '1.55', paddingTop: '3px' },
    '.cm-note-line-h3': { fontSize: '13.5px', fontWeight: '600', lineHeight: '1.6' },
    '.cm-note-line-h4': { fontSize: '12.5px', fontWeight: '600' },
    '.cm-note-line-h5': { fontSize: '12.5px', fontWeight: '600', color: darkMode ? '#9ca3af' : '#776b60' },
    '.cm-note-line-h6': { fontSize: '12.5px', fontWeight: '600', color: darkMode ? '#9ca3af' : '#776b60' },

    '.cm-note-line-quote': {
      borderLeft: `2px solid ${darkMode ? 'rgba(255,255,255,0.16)' : '#e0d7cf'}`,
      paddingLeft: '9px',
      color: darkMode ? '#9ca3af' : '#776b60',
    },

    '.cm-note-strong': { fontWeight: '600', color: darkMode ? '#f3f4f6' : '#2b2621' },
    '.cm-note-em': { fontStyle: 'italic' },
    '.cm-note-strike': { textDecoration: 'line-through', color: darkMode ? '#6b7280' : '#8f8479' },
    '.cm-note-link': { color: '#c96b50', textDecoration: 'underline', textUnderlineOffset: '2px' },
    '.cm-note-url': { color: darkMode ? '#6b7280' : '#8f8479' },
    '.cm-note-bullet': { color: '#e0855f', fontWeight: '600' },
    '.cm-note-task': { color: '#c96b50', fontWeight: '600' },
    '.cm-note-code-info': { color: darkMode ? '#9ca3af' : '#82766a' },
    '.cm-note-code': {
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
      fontSize: '11.5px',
      backgroundColor: darkMode ? 'rgba(255,255,255,0.08)' : '#f1ece7',
      borderRadius: '5px',
      padding: '1px 4px',
      color: darkMode ? '#fdc4af' : '#b85f47',
    },
  }, { dark: darkMode }),
];
