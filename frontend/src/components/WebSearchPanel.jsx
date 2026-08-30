import React, { useState } from 'react';
import { Eye, EyeOff, ExternalLink, Key, Search } from 'lucide-react';
import { useWebSearch, WEB_SEARCH_PROVIDERS } from '../contexts/WebSearchContext';
import SettingsSegmentedControl from './SettingsSegmentedControl';

const MODE_HINTS = {
  force: '每次提问都搜索网络',
  auto: '仅在需要时自动搜索网络',
  off: '仅使用文档与对话上下文',
};

/**
 * 联网搜索配置。
 *
 * 原先内联在「全局设置 > 服务」里，但它是一条检索来源，和设置中心「检索」分区
 * 的向量检索 / 知识图谱 / 检索代理是同一层的东西，放在别处会让人找不到。
 * 状态全部来自 WebSearchContext，因此不需要从外面透传。
 */
const WebSearchPanel = ({ darkMode = false }) => {
  const {
    webSearchMode, enableWebSearch, webSearchProvider, webSearchApiKey,
    webSearchBlacklist, webSearchIncludeDocumentContext,
    setWebSearchMode, setWebSearchProvider, setWebSearchApiKey,
    setWebSearchBlacklist, setWebSearchIncludeDocumentContext,
  } = useWebSearch();
  const [showApiKey, setShowApiKey] = useState(false);
  const [showBlacklist, setShowBlacklist] = useState(false);

  const currentProvider = WEB_SEARCH_PROVIDERS.find((p) => p.id === webSearchProvider) || WEB_SEARCH_PROVIDERS[0];
  const blacklistText = (webSearchBlacklist || []).join('\n');
  const handleBlacklistChange = (e) =>
    setWebSearchBlacklist(e.target.value.split('\n').map((line) => line.trim()).filter(Boolean));

  const inset = `settings-inset rounded-[16px] p-3 ${darkMode ? 'bg-[#20242a]' : 'bg-gray-50/80'}`;
  const label = `text-[12px] font-bold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`;

  return (
    <div className={`settings-card space-y-4 p-5 ${
      darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'
    }`}>
      <div>
        <SettingsSegmentedControl
          ariaLabel="联网搜索模式"
          value={webSearchMode}
          onChange={setWebSearchMode}
          options={[
            { value: 'off', label: '关闭' },
            { value: 'auto', label: '自动' },
            { value: 'force', label: '强制' },
          ]}
          buttonClassName="py-1.5 text-[12px] font-semibold text-center rounded-[12px]"
          indicatorClassName="rounded-[12px]"
        />
        <p className={`mt-2 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
          {MODE_HINTS[webSearchMode] || MODE_HINTS.off}
        </p>
      </div>

      {enableWebSearch && (
        <div className={`space-y-4 border-t pt-4 ${darkMode ? 'border-[#373b44]' : 'border-gray-100/70'}`}>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
            {WEB_SEARCH_PROVIDERS.map((provider) => {
              const isActive = webSearchProvider === provider.id;
              return (
                <button
                  key={provider.id}
                  type="button"
                  onClick={() => setWebSearchProvider(provider.id)}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-xl border p-2 transition-all ${
                    isActive
                      ? 'border-emerald-200 bg-emerald-50 text-emerald-800 shadow-sm'
                      : darkMode
                        ? 'border-[#373b44] bg-[#1d2026] text-gray-400 hover:bg-[#20242b]'
                        : 'border-gray-100 bg-gray-50 text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Search className="h-4 w-4 opacity-70" />
                  <span className="text-[13px] font-bold">{provider.name}</span>
                </button>
              );
            })}
          </div>

          {currentProvider.requiresApiKey && (
            <div className={`${inset} space-y-2`}>
              <div className="flex items-center justify-between px-1">
                <span className={label}>API 密钥</span>
                {currentProvider.url && (
                  <a
                    href={currentProvider.url}
                    target="_blank"
                    rel="noreferrer"
                    className="group flex items-center gap-1 text-[11px] text-emerald-600 hover:text-emerald-700"
                  >
                    获取 Key <ExternalLink className="h-3 w-3 transition-transform group-hover:translate-x-0.5" />
                  </a>
                )}
              </div>
              <div className="relative">
                <Key className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
                <input
                  type={showApiKey ? 'text' : 'password'}
                  value={webSearchApiKey}
                  onChange={(e) => setWebSearchApiKey(e.target.value)}
                  placeholder={`${currentProvider.name} API Key`}
                  className={`w-full flex-1 rounded-[12px] border-none py-2 pl-9 pr-10 font-mono text-[13px] shadow-sm outline-none focus:ring-2 focus:ring-emerald-500/20 ${
                    darkMode ? 'bg-[#1d2026] text-gray-100' : 'bg-white'
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  aria-label={showApiKey ? '隐藏密钥' : '显示密钥'}
                >
                  {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>
          )}

          <div className={`${inset} flex items-center justify-between gap-3`}>
            <div className="min-w-0 px-1">
              <div className={label}>附加文档上下文</div>
              <div className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                搜索时带上文件名和选中文本
              </div>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={webSearchIncludeDocumentContext}
              aria-label="附加文档上下文"
              onClick={() => setWebSearchIncludeDocumentContext(!webSearchIncludeDocumentContext)}
              className={`relative h-6 w-11 shrink-0 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/30 ${
                webSearchIncludeDocumentContext ? 'bg-emerald-500' : darkMode ? 'bg-white/15' : 'bg-gray-200'
              }`}
            >
              <span className={`absolute top-1 h-4 w-4 rounded-full bg-white transition-transform ${
                webSearchIncludeDocumentContext ? 'translate-x-6' : 'translate-x-1'
              }`} />
            </button>
          </div>

          <div className={inset}>
            <button
              type="button"
              onClick={() => setShowBlacklist(!showBlacklist)}
              aria-expanded={showBlacklist}
              className="flex w-full items-center justify-between text-left"
            >
              <div className="flex items-center gap-2 px-1">
                <span className={label}>域名黑名单配置</span>
                {webSearchBlacklist?.length > 0 && (
                  <span className={`rounded-md px-1.5 py-0.5 text-[10px] font-bold ${
                    darkMode ? 'bg-white/10 text-gray-300' : 'bg-gray-200 text-gray-600'
                  }`}>
                    {webSearchBlacklist.length}
                  </span>
                )}
              </div>
              <div className={`transform text-gray-400 transition-transform ${showBlacklist ? 'rotate-180' : ''}`}>▼</div>
            </button>
            {showBlacklist && (
              <div className="mt-3">
                <textarea
                  value={blacklistText}
                  onChange={handleBlacklistChange}
                  placeholder={'twitter.com\ninstagram.com'}
                  rows={3}
                  className={`w-full resize-none rounded-[12px] border px-3 py-2 font-mono text-[12px] shadow-sm outline-none focus:ring-2 focus:ring-emerald-500/20 ${
                    darkMode ? 'border-[#373b44] bg-[#1d2026] text-gray-100' : 'border-gray-200 bg-white'
                  }`}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default WebSearchPanel;
