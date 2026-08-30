import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, ChevronDown, ExternalLink, Type } from 'lucide-react';
import { useFontSettings, PRESET_FONTS } from '../contexts/FontSettingsContext';

/**
 * 界面字体选择（默认折叠）。
 *
 * 原先内联在「全局设置 > 显示」里，而字号又在设置中心的「界面」分区，同一类
 * 设置分裂两处。抽成独立面板后由「界面」分区直接渲染，全局设置只保留重置。
 *
 * 预设有二十多个，全部铺开会把整个「界面」分区顶到最下面。折叠态只留一行
 * 当前字体（用该字体本身渲染，一眼可辨），展开后网格再限高滚动。
 * 状态全部来自 FontSettingsContext，因此不需要从外面透传。
 */
const FontSettingsPanel = ({ darkMode = false }) => {
  const { fontFamily, customFont, setFontFamily, setCustomFont, getCurrentFontName } = useFontSettings();
  const [expanded, setExpanded] = useState(false);
  const [customFontInput, setCustomFontInput] = useState(customFont);

  const applyCustomFont = () => {
    const name = customFontInput.trim();
    if (!name) return;
    setCustomFont(name);
    setFontFamily('custom');
  };

  const activePreset = PRESET_FONTS.find((font) => font.id === fontFamily);
  const previewFontValue = fontFamily === 'custom' && customFont
    ? `"${customFont}", sans-serif`
    : activePreset?.headingValue || activePreset?.value;

  return (
    <div className={`settings-card overflow-hidden ${
      darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'
    }`}>
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded(!expanded)}
        className="settings-entry-row flex w-full items-center justify-between gap-4 px-5 py-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#D97A5D]/25"
      >
        <div className="flex min-w-0 items-center gap-3.5">
          <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-[12px] ${
            darkMode ? 'bg-white/[0.055] text-gray-400' : 'bg-gray-100 text-gray-500'
          }`}>
            <Type size={17} strokeWidth={2.1} />
          </div>
          <div className="min-w-0">
            <div className={`text-[13px] font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
              界面字体
            </div>
            <p className={`mt-1 truncate text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
              当前：
              <span
                className={`font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}
                style={{ fontFamily: previewFontValue }}
              >
                {getCurrentFontName()}
              </span>
            </p>
          </div>
        </div>
        <ChevronDown className={`h-4 w-4 shrink-0 text-gray-400 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className={`space-y-3 border-t px-5 pb-5 pt-4 ${darkMode ? 'border-[#373b44]' : 'border-gray-100'}`}>
              {/* 预设过多，网格自己限高滚动，避免展开后把下方设置顶出视野 */}
              <div className="custom-scrollbar -mx-1 max-h-[292px] overflow-y-auto px-1 pb-1">
                <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
                  {PRESET_FONTS.map((font) => {
                    const isActive = fontFamily === font.id;
                    return (
                      <button
                        key={font.id}
                        type="button"
                        onClick={() => setFontFamily(font.id)}
                        aria-pressed={isActive}
                        className={`settings-card settings-card-interactive relative p-3.5 text-left ${
                          isActive
                            ? 'accent-surface'
                            : darkMode
                              ? 'settings-card-dark bg-[#1d2026] border-[#373b44]'
                              : 'bg-white border-gray-200/90'
                        }`}
                      >
                        <div className="mb-2.5 flex items-center justify-between">
                          <div className={`flex h-4 w-4 flex-shrink-0 items-center justify-center rounded-full transition-colors ${
                            isActive ? 'accent-control' : darkMode ? 'bg-white/15' : 'bg-gray-200'
                          }`}>
                            {isActive && <Check className="h-2.5 w-2.5 text-[#B85F47]" strokeWidth={3} />}
                          </div>
                          {isActive && (
                            <span className="accent-surface flex-shrink-0 whitespace-nowrap rounded-full px-2 py-0.5 text-[10px] font-bold uppercase">
                              Default
                            </span>
                          )}
                        </div>
                        <div
                          className={`mb-1 line-clamp-2 min-h-[34px] text-[14px] font-bold leading-tight ${
                            darkMode && !isActive ? 'text-gray-100' : 'text-gray-900'
                          }`}
                          style={{ fontFamily: font.headingValue || font.value }}
                        >
                          {font.name}
                        </div>
                        <div
                          className={`line-clamp-2 min-h-[26px] pr-1 text-[11px] leading-tight ${
                            darkMode && !isActive ? 'text-gray-400' : 'text-gray-500'
                          }`}
                          style={{ fontFamily: font.bodyValue || font.value }}
                        >
                          {font.description}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className={`settings-inset flex items-center justify-between gap-3 rounded-[16px] p-2 pl-4 ${
                darkMode ? 'bg-[#20242a]' : 'bg-gray-50/80'
              }`}>
                <div className="min-w-0 flex-1">
                  <div className={`truncate text-[12px] font-bold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    自定义 Google 字体
                  </div>
                  <div className={`truncate text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                    输入 Google Font 名称即可加载
                  </div>
                </div>
                <div className={`flex w-[200px] flex-shrink-0 items-center gap-1.5 rounded-2xl border py-1.5 pl-3 pr-1.5 sm:w-[230px] ${
                  darkMode ? 'border-[#3b4049] bg-[#1d2026]' : 'border-gray-200/70 bg-white'
                }`}>
                  <ExternalLink className="h-3.5 w-3.5 flex-shrink-0 text-gray-400" />
                  <input
                    type="text"
                    value={customFontInput}
                    onChange={(e) => setCustomFontInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') applyCustomFont(); }}
                    placeholder="例如 Inter"
                    className={`min-w-0 flex-1 border-none bg-transparent text-[13px] font-medium outline-none placeholder-gray-400 ${
                      darkMode ? 'text-gray-100' : ''
                    }`}
                  />
                  <button
                    type="button"
                    onClick={applyCustomFont}
                    className="accent-surface flex-shrink-0 rounded-[12px] px-3 py-1 text-[12px] font-bold transition-all"
                  >
                    应用
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default FontSettingsPanel;
