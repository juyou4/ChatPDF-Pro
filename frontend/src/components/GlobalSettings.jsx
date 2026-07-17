import React, { useState } from 'react';
import { ChevronLeft, Type, ZoomIn, RotateCcw, Download, Upload, Check, Brain, Globe, ExternalLink, Eye, EyeOff, CheckCircle2, Search, Key } from 'lucide-react';
import MemoryPanel from './MemoryPanel';
import { useFontSettings, PRESET_FONTS } from '../contexts/FontSettingsContext';
import { useChatParams } from '../contexts/ChatParamsContext';
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';
import { useWebSearch, WEB_SEARCH_PROVIDERS } from '../contexts/WebSearchContext';
import { useReadingSettings } from '../contexts/ReadingSettingsContext';
import { motion, AnimatePresence } from 'framer-motion';
import SettingsSegmentedControl from './SettingsSegmentedControl';

const GlobalSettings = ({ isOpen, onClose }) => {
    const {
        fontFamily, customFont, globalScale, setFontFamily, setCustomFont, setGlobalScale, resetFontSettings
    } = useFontSettings();
    // 检索增强调优已上移到设置中心一级「检索」tab；此处仅保留重置能力
    const { enableMemory, setEnableMemory, resetChatParams } = useChatParams();
    const [activeSection, setActiveSection] = useState('display');
    const { exportSettings, importSettings } = useGlobalSettings();
    // 阅读相关设置已上移到设置中心一级「阅读」tab；此处仅保留重置能力
    const { resetReadingSettings } = useReadingSettings();

    const resetSettings = () => { resetFontSettings(); resetChatParams(); resetWebSearch(); resetReadingSettings(); };

    const [customFontInput, setCustomFontInput] = useState(customFont);
    const [showMemoryPanel, setShowMemoryPanel] = useState(false);
    const [showImportDialog, setShowImportDialog] = useState(false);
    const [importText, setImportText] = useState('');
    const [showApiKey, setShowApiKey] = useState(false);
    const [showBlacklist, setShowBlacklist] = useState(false);

    const {
        enableWebSearch, webSearchProvider, webSearchApiKey, webSearchBlacklist,
        setEnableWebSearch, setWebSearchProvider, setWebSearchApiKey, setWebSearchBlacklist,
        resetWebSearch,
    } = useWebSearch();
    const currentSearchProvider = WEB_SEARCH_PROVIDERS.find(p => p.id === webSearchProvider) || WEB_SEARCH_PROVIDERS[0];
    const needsApiKey = currentSearchProvider.requiresApiKey;
    const blacklistText = (webSearchBlacklist || []).join('\n');
    const handleBlacklistChange = (e) => setWebSearchBlacklist(e.target.value.split('\n').map(l => l.trim()).filter(Boolean));

    const scalePresets = [
        { label: '75%', value: 0.75 }, { label: '100%', value: 1.0 },
        { label: '125%', value: 1.25 }, { label: '150%', value: 1.5 },
    ];

    const handleExport = () => {
        const json = exportSettings();
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chatpdf-settings-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const handleImport = () => {
        if (importSettings(importText)) { alert('✅ 设置导入成功！'); setShowImportDialog(false); setImportText(''); }
        else alert('❌ 导入失败，请检查 JSON 格式是否正确');
    };

    const handleFileImport = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                if (importSettings(event.target.result)) alert('✅ 设置导入成功！');
                else alert('❌ 导入失败，请检查文件格式');
            };
            reader.readAsText(file);
        }
    };

    const applyCustomFont = () => {
        if (customFontInput.trim()) {
            setCustomFont(customFontInput.trim());
            setFontFamily('custom');
        }
    };

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="fixed inset-0 bg-slate-950/25 z-50 flex items-center justify-center p-4 transition-all"
                    onClick={onClose}
                >
                    <motion.div
                        initial={{ scale: 0.95, opacity: 0, y: 15 }} animate={{ scale: 1, opacity: 1, y: 0 }} exit={{ scale: 0.95, opacity: 0, y: 10 }}
                        transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                        className="settings-solid settings-shell w-full max-w-[700px] max-h-[92vh] bg-[#f6f7f9] border border-white/80 overflow-hidden flex flex-col relative"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* 头部 */}
                        <div className="settings-chrome flex items-center px-7 py-5 sticky top-0 border-b border-gray-200 z-10">
                            <div className="flex items-center gap-3">
                                <button onClick={onClose} className="p-2 -ml-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-colors" title="返回设置中心" aria-label="返回设置中心">
                                    <ChevronLeft className="w-5 h-5" />
                                </button>
                                <div className="w-9 h-9 flex items-center justify-center text-[#ed8c68]">
                                    <Type className="w-5 h-5 fill-current opacity-20" />
                                    <Type className="w-5 h-5 absolute" />
                                </div>
                                <h2 className="text-[17px] font-bold text-gray-900 tracking-tight">全局设置</h2>
                            </div>
                        </div>

                        {/* 「阅读」分区已上移到设置中心一级 tab，避免同一设置出现在两处 */}
                        <div className="settings-segment relative grid grid-cols-3 mx-7 mb-4 p-1 rounded-[20px] flex-shrink-0" role="tablist" aria-label="全局设置分类">
                            <span
                                aria-hidden="true"
                                className="settings-segment-indicator absolute left-1 top-1 bottom-1 rounded-[16px] transition-transform duration-[320ms] ease-[cubic-bezier(0.4,0,0.2,1)] motion-reduce:transition-none"
                                style={{
                                    width: 'calc((100% - 0.5rem) / 3)',
                                    transform: `translateX(${['display', 'services', 'advanced'].indexOf(activeSection) * 100}%)`,
                                }}
                            />
                            {[
                                { id: 'display', label: '显示' },
                                { id: 'services', label: '服务' },
                                { id: 'advanced', label: '高级' },
                            ].map((section) => (
                                <button
                                    key={section.id}
                                    type="button"
                                    role="tab"
                                    aria-selected={activeSection === section.id}
                                    onClick={() => setActiveSection(section.id)}
                                    className={`relative z-10 min-w-0 rounded-[16px] px-2 py-2 text-[12px] font-semibold transition-colors ${
                                        activeSection === section.id
                                            ? 'text-[#ed8c68]'
                                            : 'text-gray-500 hover:text-gray-800 hover:bg-gray-200/70'
                                    }`}
                                >
                                    {section.label}
                                </button>
                            ))}
                        </div>

                        <motion.div
                            key={activeSection}
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.18, ease: 'easeOut' }}
                            className="flex-1 overflow-y-auto px-7 pb-[100px] pt-1 space-y-6 custom-scrollbar"
                        >
                            {activeSection === 'display' && (
                            <>
                            {/* Font Settings - Cards Grid */}
                            <div className="space-y-4">
                                <div>
                                    <h3 className="text-[17px] font-bold text-gray-900">全局字体设置</h3>
                                    <p className="text-[13px] text-gray-500 mt-1">选择用于界面与助手的标题、正文配对字体。</p>
                                </div>

                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                    {PRESET_FONTS.map((font) => (
                                        <button
                                            key={font.id} onClick={() => setFontFamily(font.id)}
                                            aria-pressed={fontFamily === font.id}
                                            className={`settings-card settings-card-interactive relative p-4 rounded-[20px] text-left ${
                                                fontFamily === font.id
                                                    ? 'accent-surface'
                                                    : 'bg-white'
                                            }`}
                                        >
                                            {/* 选择圆点指示器 */}
                                            <div className="mb-3 flex items-center justify-between">
                                                <div className={`w-4 h-4 rounded-full flex items-center justify-center transition-colors flex-shrink-0 ${fontFamily === font.id ? 'accent-control' : 'bg-gray-200'}`}>
                                                    {fontFamily === font.id && <Check className="w-2.5 h-2.5 text-[#B85F47]" strokeWidth={3} />}
                                                </div>
                                                {fontFamily === font.id && <span className="accent-surface text-[10px] uppercase font-bold px-2 py-0.5 rounded-full whitespace-nowrap flex-shrink-0">Default</span>}
                                            </div>
                                            <div
                                                className="font-bold text-[15px] leading-tight min-h-[36px] mb-1 text-gray-900 line-clamp-2"
                                                style={{ fontFamily: font.headingValue || font.value }}
                                            >
                                                {font.name}
                                            </div>
                                            <div className="text-[11px] text-gray-500 leading-tight pr-2 min-h-[28px] line-clamp-2" style={{ fontFamily: font.bodyValue || font.value }}>
                                                {font.description}
                                                {!font.description && font.id === 'inter' && '严谨、几何感的无衬线体。'}
                                                {!font.description && font.id === 'noto-sans-sc' && '完美的 CJK 多语言字重协调。'}
                                                {!font.description && font.id === 'outfit' && '现代科技感的规整骨架。'}
                                                {!font.description && font.id === 'lato' && '清爽干净的高级阅读体验。'}
                                                {!font.description && font.id === 'lora' && '古典、优雅的衬线字体。'}
                                            </div>
                                        </button>
                                    ))}
                                </div>

                                {/* 自定义 Google 字体 Input */}
                                <div className="settings-card bg-white rounded-[20px] p-2 pl-4 mt-2 flex items-center justify-between gap-3">
                                    <div className="flex-1 min-w-0">
                                        <div className="text-[13px] font-bold text-gray-800 truncate">自定义 Google 字体</div>
                                        <div className="text-[11px] text-gray-500 truncate">输入任何 Google Font 名称以自动加载。</div>
                                    </div>
                                    <div className="flex items-center gap-1.5 bg-white pl-3 pr-1.5 py-1.5 rounded-2xl border border-gray-200/70 w-[200px] sm:w-[240px] flex-shrink-0">
                                        <ExternalLink className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
                                        <input
                                            type="text" value={customFontInput} onChange={(e) => setCustomFontInput(e.target.value)}
                                            placeholder="输入字体名称，例如 Inter"
                                            className="flex-1 min-w-0 text-[13px] border-none outline-none bg-transparent placeholder-gray-400 font-medium"
                                        />
                                        <button
                                            onClick={applyCustomFont}
                                            className="accent-surface px-3 py-1 rounded-[12px] text-[12px] font-bold transition-all flex-shrink-0"
                                        >
                                            应用
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {/* Scale/Zoom Settings */}
                            <div className="settings-card bg-white p-5 border border-gray-200/90">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-2">
                                        <ZoomIn className="w-4 h-4 text-gray-400" />
                                        <span className="text-[14px] font-medium text-gray-800">界面缩放倍率</span>
                                    </div>
                                    <div className="px-3 py-1 bg-gray-50 rounded-[12px] border border-gray-100 text-[13px] font-bold text-gray-700">
                                        {Math.round(globalScale * 100)}%
                                    </div>
                                </div>
                                <div className="space-y-4">
                                    <div className="px-1 py-1">
                                        <input
                                            type="range" min="0.5" max="2.0" step="0.05" value={globalScale} onChange={(e) => setGlobalScale(parseFloat(e.target.value))}
                                            className="w-full h-[6px] rounded-full appearance-none bg-gray-100 cursor-pointer"
                                            style={{ background: `linear-gradient(to right, #ed8c68 0%, #ed8c68 ${((globalScale - 0.5) / 1.5) * 100}%, #F3F4F6 ${((globalScale - 0.5) / 1.5) * 100}%, #F3F4F6 100%)` }}
                                        />
                                    </div>
                                    <SettingsSegmentedControl
                                        ariaLabel="界面缩放预设"
                                        value={scalePresets.find((preset) => Math.abs(globalScale - preset.value) < 0.01)?.value ?? null}
                                        onChange={setGlobalScale}
                                        options={scalePresets}
                                        buttonClassName="py-1.5 text-[12px] font-medium text-center rounded-[9px]"
                                        indicatorClassName="rounded-[9px]"
                                    />
                                </div>
                            </div>
                            </>
                            )}

                            {activeSection === 'services' && (
                            <>
                            {/* Memory Settings */}
                            <div className="settings-card bg-white p-5 border border-gray-200/90">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-blue-50 text-blue-500 flex items-center justify-center">
                                            <Brain className="w-4 h-4" />
                                        </div>
                                        <div>
                                            <h3 className="text-[14px] font-bold text-gray-900">智能记忆系统</h3>
                                            <p className="text-[12px] text-gray-500">AI 将记住你的对话偏好和重要信息</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <button onClick={() => setShowMemoryPanel(true)} className="text-[12px] font-bold text-blue-600 bg-blue-50 hover:bg-blue-100 px-3 py-1.5 rounded-xl transition-colors">
                                            管理记忆
                                        </button>
                                        <ToggleSwitch checked={enableMemory} onChange={setEnableMemory} color="#3b82f6" />
                                    </div>
                                </div>
                            </div>

                            {/* Web Search Settings */}
                            <div className="settings-card bg-white p-5 border border-gray-200/90 space-y-4">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-emerald-50 text-emerald-500 flex items-center justify-center">
                                            <Globe className="w-4 h-4" />
                                        </div>
                                        <div>
                                            <h3 className="text-[14px] font-bold text-gray-900">联网搜索</h3>
                                            <p className="text-[12px] text-gray-500">对话时自动搜索网络补充信息</p>
                                        </div>
                                    </div>
                                    <ToggleSwitch checked={enableWebSearch} onChange={setEnableWebSearch} color="#10b981" />
                                </div>

                                {enableWebSearch && (
                                    <div className="pt-3 border-t border-gray-100/50 space-y-4">
                                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                                            {WEB_SEARCH_PROVIDERS.map((provider) => (
                                                <button
                                                    key={provider.id} onClick={() => setWebSearchProvider(provider.id)}
                                                    className={`flex-1 p-2 rounded-xl flex items-center justify-center gap-2 transition-all border ${
                                                        webSearchProvider === provider.id
                                                            ? 'bg-emerald-50 border-emerald-200 text-emerald-800 shadow-sm'
                                                            : 'bg-gray-50 border-gray-100 text-gray-600 hover:bg-gray-100'
                                                    }`}
                                                >
                                                    <Search className="w-4 h-4 opacity-70" />
                                                    <span className="text-[13px] font-bold">{provider.name}</span>
                                                </button>
                                            ))}
                                        </div>

                                        {needsApiKey && (
                                            <div className="bg-gray-50/80 p-3 rounded-[16px] border border-gray-100/80 space-y-2">
                                                <div className="flex items-center justify-between px-1">
                                                    <span className="text-[12px] font-bold text-gray-700">API Key API 密钥</span>
                                                    {currentSearchProvider.url && (
                                                        <a href={currentSearchProvider.url} target="_blank" rel="noreferrer" className="text-[11px] text-emerald-600 group flex items-center gap-1 hover:text-emerald-700">
                                                            获取 Key <ExternalLink className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                                                        </a>
                                                    )}
                                                </div>
                                                <div className="relative">
                                                    <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                                    <input
                                                        type={showApiKey ? 'text' : 'password'} value={webSearchApiKey} onChange={(e) => setWebSearchApiKey(e.target.value)}
                                                        placeholder={`${currentSearchProvider.name} API Key`}
                                                        className="w-full flex-1 text-[13px] font-mono border-none outline-none bg-white py-2 pl-9 pr-10 rounded-[12px] shadow-sm focus:ring-2 focus:ring-emerald-500/20"
                                                    />
                                                    <button onClick={() => setShowApiKey(!showApiKey)} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                                                        {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                                    </button>
                                                </div>
                                            </div>
                                        )}

                                        <div className="bg-gray-50/80 p-3 rounded-[16px] border border-gray-100/80">
                                            <button onClick={() => setShowBlacklist(!showBlacklist)} className="w-full flex items-center justify-between text-left">
                                                <div className="flex items-center gap-2 px-1">
                                                    <span className="text-[12px] font-bold text-gray-700">域名黑名单配置</span>
                                                    {webSearchBlacklist?.length > 0 && <span className="bg-gray-200 text-gray-600 text-[10px] px-1.5 py-0.5 rounded-md font-bold">{webSearchBlacklist.length}</span>}
                                                </div>
                                                <div className={`transform transition-transform text-gray-400 ${showBlacklist ? 'rotate-180' : ''}`}>▼</div>
                                            </button>
                                            {showBlacklist && (
                                                <div className="mt-3">
                                                    <textarea
                                                        value={blacklistText} onChange={handleBlacklistChange} placeholder="twitter.com\ninstagram.com" rows={3}
                                                        className="w-full text-[12px] font-mono bg-white border border-gray-200 rounded-[12px] px-3 py-2 outline-none focus:ring-2 focus:ring-emerald-500/20 resize-none shadow-sm"
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                            </>
                            )}

                            {/* Data Mgmt — 检索增强调优已上移到设置中心一级「检索」tab */}
                            {activeSection === 'advanced' && (
                            <>
                            <div className="grid grid-cols-4 gap-3 pt-2">
                                {[
                                    { icon: RotateCcw, label: '恢复默认', onClick: () => { if(confirm('确定?')) { resetSettings(); setCustomFontInput(''); } }, color: 'text-red-500 hover:bg-red-50 hover:border-red-200' },
                                    { icon: Download, label: '导出配置', onClick: handleExport, color: 'text-gray-600 hover:bg-gray-100' },
                                    { icon: Upload, label: '导入文件', action: 'file', color: 'text-gray-600 hover:bg-gray-100' },
                                    { icon: Type, label: '导入文本', onClick: () => setShowImportDialog(true), color: 'text-gray-600 hover:bg-gray-100' },
                                ].map((btn, i) => (
                                    btn.action === 'file' ? (
                                        <label key={i} className={`settings-card settings-card-interactive flex flex-col items-center justify-center gap-1.5 p-3 rounded-[16px] bg-white cursor-pointer ${btn.color}`}>
                                            <btn.icon className="w-4 h-4 opacity-80" /> <span className="text-[11px] font-bold">{btn.label}</span>
                                            <input type="file" accept=".json" onChange={handleFileImport} className="hidden" />
                                        </label>
                                    ) : (
                                        <button key={i} onClick={btn.onClick} className={`settings-card settings-card-interactive flex flex-col items-center justify-center gap-1.5 p-3 rounded-[16px] bg-white ${btn.color}`}>
                                            <btn.icon className="w-4 h-4 opacity-80" /> <span className="text-[11px] font-bold">{btn.label}</span>
                                        </button>
                                    )
                                ))}
                            </div>
                            </>
                            )}
                        </motion.div>

                        {/* Floating Bottom Action */}
                        <div className="settings-chrome absolute inset-x-0 bottom-0 p-4 border-t border-gray-200 flex justify-center pointer-events-none">
                            <button
                                onClick={onClose}
                                className="accent-surface pointer-events-auto font-bold text-[14px] py-3 px-8 flex items-center justify-center gap-2 rounded-[14px] transition-colors"
                            >
                                <CheckCircle2 className="w-5 h-5" />
                                完成并返回
                            </button>
                        </div>
                    </motion.div>

                    {/* 文本导入附赠弹窗保持基本干净即可 */}
                    {showImportDialog && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="fixed inset-0 bg-slate-950/45 z-[60] flex items-center justify-center p-4" onClick={() => setShowImportDialog(false)}>
                            <div className="bg-white border border-gray-200 rounded-[18px] p-6 max-w-sm w-full shadow-[0_24px_60px_-18px_rgba(15,23,42,0.35)]" onClick={(e) => e.stopPropagation()}>
                                <h3 className="text-[15px] font-bold mb-3">粘贴 JSON 配置</h3>
                                <textarea value={importText} onChange={(e) => setImportText(e.target.value)} className="w-full h-32 p-3 bg-gray-50 border border-gray-200 rounded-[16px] text-xs font-mono outline-none focus:ring-2 focus:ring-[#ed8c68]/20 resize-none mb-4" />
                                <div className="flex gap-2">
                                    <button onClick={() => setShowImportDialog(false)} className="flex-1 py-2 bg-gray-100 text-gray-700 rounded-xl text-sm font-bold">取消</button>
                                    <button onClick={handleImport} className="accent-surface flex-1 py-2 rounded-xl text-sm font-bold">导入</button>
                                </div>
                            </div>
                        </motion.div>
                    )}
                    <MemoryPanel isOpen={showMemoryPanel} onClose={() => setShowMemoryPanel(false)} />
                </motion.div>
            )}
        </AnimatePresence>
    );
};

/* 子组件 */
const ToggleSwitch = ({ checked, onChange, color = '#ed8c68' }) => (
    <button onClick={() => onChange(!checked)} className="relative w-[42px] h-[24px] rounded-full transition-colors duration-200 outline-none flex-shrink-0" style={{ backgroundColor: checked ? color : '#e5e7eb' }}>
        <div className={`absolute top-[2px] left-[2px] w-[20px] h-[20px] bg-white rounded-full shadow-sm transition-transform duration-200 ${checked ? 'translate-x-[18px]' : ''}`} />
    </button>
);

const KeyIcon = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4"/><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/></svg>
);

export default GlobalSettings;
