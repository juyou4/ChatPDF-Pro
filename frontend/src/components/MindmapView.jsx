import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Map, Copy } from 'lucide-react';
import StreamingMarkdown from './StreamingMarkdown';

/**
 * 思维导图面板：以可折叠 Markdown heading 形式展示思维导图
 * 
 * Props:
 *   markdown: string - Markdown heading 格式的思维导图文本
 */
export default function MindmapView({ markdown }) {
  const [collapsed, setCollapsed] = useState(true);
  const [copied, setCopied] = useState(false);

  if (!markdown) return null;

  const handleCopy = async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(markdown);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  };

  return (
    <div className="mt-2 ml-2">
      <button
        onClick={() => setCollapsed(prev => !prev)}
        className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors mb-1.5"
      >
        {collapsed
          ? <ChevronRight className="w-3.5 h-3.5" />
          : <ChevronDown className="w-3.5 h-3.5" />
        }
        <Map className="w-3.5 h-3.5" />
        <span className="font-medium">思维导图</span>
      </button>
      {!collapsed && (
        <div className="relative border border-gray-200 rounded-lg p-3 bg-gray-50/50 max-w-lg">
          <button
            onClick={handleCopy}
            className="absolute top-2 right-2 p-1 rounded hover:bg-gray-200 text-gray-400 hover:text-gray-600 transition-colors"
            title="复制 Markdown"
          >
            {copied
              ? <svg className="w-3.5 h-3.5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
              : <Copy className="w-3.5 h-3.5" />
            }
          </button>
          <div className="text-xs mindmap-content">
            <StreamingMarkdown content={markdown} isStreaming={false} />
          </div>
        </div>
      )}
    </div>
  );
}
