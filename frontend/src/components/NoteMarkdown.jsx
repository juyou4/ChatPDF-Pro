import React, { memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';

const NOTE_REMARK_PLUGINS = [
  remarkGfm,
  [remarkMath, { singleDollarTextMath: true }],
];

const NOTE_REHYPE_PLUGINS = [
  [rehypeKatex, { strict: false, trust: false, output: 'html' }],
  rehypeHighlight,
];

const getSafeNoteHref = (href) => {
  const value = String(href || '').trim();
  if (!value) return '';
  if (value.startsWith('#')) return value;
  try {
    const url = new URL(value);
    return ['http:', 'https:', 'mailto:'].includes(url.protocol) ? url.href : '';
  } catch {
    return '';
  }
};

function NoteMarkdown({ content = '', darkMode = false, className = '' }) {
  return (
    <div className={`note-markdown ${darkMode ? 'note-markdown--dark' : ''} ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={NOTE_REMARK_PLUGINS}
        rehypePlugins={NOTE_REHYPE_PLUGINS}
        components={{
          a: ({ href, children }) => {
            const safeHref = getSafeNoteHref(href);
            if (!safeHref) return <span>{children}</span>;
            const external = !safeHref.startsWith('#');
            return (
              <a
                href={safeHref}
                target={external ? '_blank' : undefined}
                rel={external ? 'noreferrer noopener' : undefined}
                onClick={(event) => event.stopPropagation()}
              >
                {children}
              </a>
            );
          },
          img: ({ src, alt, title }) => (
            <img src={src} alt={alt || '笔记图片'} title={title} loading="lazy" />
          ),
          table: ({ children }) => (
            <div className="note-markdown-table-wrap">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {String(content || '')}
      </ReactMarkdown>
    </div>
  );
}

export default memo(NoteMarkdown);
