import React, { useState } from 'react';
import { X, Type, ZoomIn, RotateCcw, Download, Upload, Check, Brain, Globe, ExternalLink, Eye, EyeOff, CheckCircle2, Search, Key, Sparkles } from 'lucide-react';
import MemoryPanel from './MemoryPanel';
import { useFontSettings, PRESET_FONTS } from '../contexts/FontSettingsContext';
import { useChatParams } from '../contexts/ChatParamsContext';
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';
import { useWebSearch, WEB_SEARCH_PROVIDERS } from '../contexts/WebSearchContext';
import { motion, AnimatePresence } from 'framer-motion';

const GlobalSettings = ({ isOpen, onClose }) => {
    const {
        fontFamily, customFont, globalScale, setFontFamily, setCustomFont, setGlobalScale, resetFontSettings
    } = useFontSettings();
    const {
        enableMemory, setEnableMemory, resetChatParams,
        overrideNumericTable, setOverrideNumericTable,
        overrideAnswerCritic, setOverrideAnswerCritic,
        overrideLLMQueryRewrite, setOverrideLLMQueryRewrite,
        overrideBM25Synonyms, setOverrideBM25Synonyms,
        cheapModel, setCheapModel,
        cheapModelProvider, setCheapModelProvider,
    } = useChatParams();
    const [showRetrievalTuning, setShowRetrievalTuning] = useState(false);
    const { exportSettings, importSettings } = useGlobalSettings();

    const resetSettings = () => { resetFontSettings(); resetChatParams(); resetWebSearch(); };

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
                    className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 flex items-center justify-center p-4 transition-all"
                    onClick={onClose}
                >
                    <motion.div
                        initial={{ scale: 0.95, opacity: 0, y: 15 }} animate={{ scale: 1, opacity: 1, y: 0 }} exit={{ scale: 0.95, opacity: 0, y: 10 }}
                        transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                        className="w-full max-w-[700px] max-h-[92vh] bg-[#fbfbfc] shadow-[0_24px_60px_-15px_rgba(0,0,0,0.1)] rounded-[32px] overflow-hidden flex flex-col relative"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* 头部 */}
                        <div className="flex items-center justify-between px-7 py-5 sticky top-0 bg-[#fbfbfc]/90 backdrop-blur-md z-10">
                            <div className="flex items-center gap-3">
                                <div className="w-9 h-9 flex items-center justify-center text-[#8871e4]">
                                    <Type className="w-5 h-5 fill-current opacity-20" />
                                    <Type className="w-5 h-5 absolute" />
                                </div>
                                <h2 className="text-[17px] font-bold text-gray-900 tracking-tight">全局设置</h2>
                            </div>
                            <button onClick={onClose} className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="flex-1 overflow-y-auto px-7 pb-[100px] pt-1 space-y-6 custom-scrollbar">
                            {/* Font Settings - Cards Grid */}
                            <div className="space-y-4">
                                <div>
                                    <h3 className="text-[17px] font-bold text-gray-900">全局字体设置</h3>
                                    <p className="text-[13px] text-gray-500 mt-1">选择用于界面与助手的全局基础字体。</p>
                                </div>

                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                    {PRESET_FONTS.map((font) => (
                                        <button
                                            key={font.id} onClick={() => setFontFamily(font.id)}
                                            className={`relative p-4 rounded-[20px] text-left transition-all border ${
                                                fontFamily === font.id
                                                    ? 'border-[#8871e4] bg-[#8871e4]/5 shadow-[0_2px_10px_-2px_rgba(136,113,228,0.15)]'
                                                    : 'border-transparent bg-white shadow-sm hover:border-gray-200'
                                            }`}
                                        >
                                            {/* 选择圆点指示器 */}
                                            <div className="mb-3 flex items-center justify-between">
                                                <div className={`w-4 h-4 rounded-full flex items-center justify-center transition-colors flex-shrink-0 ${fontFamily === font.id ? 'bg-[#8871e4]' : 'bg-gray-200'}`}>
                                                    {fontFamily === font.id && <Check className="w-2.5 h-2.5 text-white" strokeWidth={3} />}
                                                </div>
                                                {fontFamily === font.id && <span className="text-[10px] uppercase font-bold text-[#8871e4] bg-[#8871e4]/10 px-2 py-0.5 rounded-full whitespace-nowrap flex-shrink-0">Default</span>}
                                            </div>
                                            <div className="font-bold text-[15px] mb-1 text-gray-900 truncate">{font.name}</div>
                                            <div className="text-[11px] text-gray-500 leading-tight pr-2" style={{ fontFamily: font.value }}>
                                                {font.id === 'inter' && '严谨、几何感的无衬线体。'}
                                                {font.id === 'noto-sans-sc' && '完美的 CJK 多语言字重协调。'}
                                                {font.id === 'outfit' && '现代科技感的规整骨架。'}
                                                {font.id === 'lato' && '清爽干净的高级阅读体验。'}
                                                {font.id === 'lora' && '古典、优雅的衬线字体。'}
                                            </div>
                                        </button>
                                    ))}
                                </div>

                                {/* 自定义 Google 字体 Input */}
                                <div className="bg-gray-100/60 rounded-[20px] p-2 pl-4 mt-2 flex items-center justify-between gap-3 border border-gray-200/50">
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
                                            className={`px-3 py-1 rounded-[12px] text-[12px] font-bold transition-all flex-shrink-0 ${
                                                fontFamily === 'custom' ? 'bg-[#8871e4] text-white' : 'bg-[#8871e4] text-white hover:bg-[#725ec3]'
                                            }`}
                                        >
                                            Apply
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {/* Scale/Zoom Settings */}
                            <div className="bg-white rounded-[24px] p-5 shadow-sm border border-gray-100/80">
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
                                            style={{ background: `linear-gradient(to right, #8871e4 0%, #8871e4 ${((globalScale - 0.5) / 1.5) * 100}%, #F3F4F6 ${((globalScale - 0.5) / 1.5) * 100}%, #F3F4F6 100%)` }}
                                        />
                                    </div>
                                    <div className="flex gap-2">
                                        {scalePresets.map((preset) => (
                                            <button
                                                key={preset.value} onClick={() => setGlobalScale(preset.value)}
                                                className={`flex-1 py-1.5 text-[12px] font-medium rounded-xl transition-all ${
                                                    Math.abs(globalScale - preset.value) < 0.01
                                                        ? 'bg-[#8871e4]/10 text-[#8871e4] border border-[#8871e4]/30'
                                                        : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-transparent'
                                                }`}
                                            >
                                                {preset.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Memory Settings */}
                            <div className="bg-white rounded-[24px] p-5 shadow-sm border border-gray-100/80">
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
                            <div className="bg-white rounded-[24px] p-5 shadow-sm border border-gray-100/80 space-y-4">
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

                            {/* Retrieval Tuning — 检索增强调优 */}
                            <div className="bg-white rounded-[24px] p-5 shadow-sm border border-gray-100/80 space-y-4">
                                <button onClick={() => setShowRetrievalTuning(!showRetrievalTuning)} className="w-full flex items-center justify-between text-left">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-violet-50 text-violet-500 flex items-center justify-center">
                                            <Sparkles className="w-4 h-4" />
                                        </div>
                                        <div>
                                            <h3 className="text-[14px] font-bold text-gray-900">检索增强调优</h3>
                                            <p className="text-[12px] text-gray-500">对当前会话覆盖后端检索开关，不持久化到后端 config</p>
                                        </div>
                                    </div>
                                    <div className={`transform transition-transform text-gray-400 ${showRetrievalTuning ? 'rotate-180' : ''}`}>▼</div>
                                </button>

                                {showRetrievalTuning && (
                                    <div className="pt-3 border-t border-gray-100/50 space-y-3">
                                        <TriStateToggle
                                            title="numeric_table 专项增强"
                                            desc="表格数值比较类查询（如「第二好的方法」「Table 7 DiffuLT」）的专项检索增强"
                                            value={overrideNumericTable}
                                            onChange={setOverrideNumericTable}
                                        />
                                        <TriStateToggle
                                            title="BM25 同义词扩展"
                                            desc="查询时自动扩展同义词，提升召回率"
                                            value={overrideBM25Synonyms}
                                            onChange={setOverrideBM25Synonyms}
                                        />
                                        <TriStateToggle
                                            title="LLM 查询改写"
                                            desc="多轮对话中用 LLM 消解指代（代词/省略），长查询自动跳过"
                                            value={overrideLLMQueryRewrite}
                                            onChange={setOverrideLLMQueryRewrite}
                                        />
                                        <TriStateToggle
                                            title="答案自审"
                                            desc="回答结束后用 cheap model 检测幻觉；会增加 1-3s 延迟"
                                            value={overrideAnswerCritic}
                                            onChange={setOverrideAnswerCritic}
                                        />

                                        {/* Cheap Model 配置 */}
                                        <div className="bg-gray-50/80 p-3 rounded-[16px] border border-gray-100/80 space-y-2">
                                            <div className="flex items-center justify-between px-1">
                                                <span className="text-[12px] font-bold text-gray-700">辅助模型（双模型策略）</span>
                                                <span className="text-[10px] text-gray-500">为空则跟随后端默认</span>
                                            </div>
                                            <p className="text-[11px] text-gray-500 px-1">
                                                用于非核心 LLM 任务（查询改写 / 追问建议 / 自动命名 / 答案自审）
                                            </p>
                                            <div className="grid grid-cols-2 gap-2">
                                                <input
                                                    type="text"
                                                    value={cheapModelProvider || ''}
                                                    onChange={(e) => setCheapModelProvider(e.target.value)}
                                                    placeholder="provider (如 openai)"
                                                    className="text-[12px] font-mono bg-white border border-gray-200 rounded-[12px] px-3 py-2 outline-none focus:ring-2 focus:ring-violet-500/20 shadow-sm"
                                                />
                                                <input
                                                    type="text"
                                                    value={cheapModel || ''}
                                                    onChange={(e) => setCheapModel(e.target.value)}
                                                    placeholder="model (如 gpt-4o-mini)"
                                                    className="text-[12px] font-mono bg-white border border-gray-200 rounded-[12px] px-3 py-2 outline-none focus:ring-2 focus:ring-violet-500/20 shadow-sm"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Data Mgmt */}
                            <div className="grid grid-cols-4 gap-3 pt-2">
                                {[
                                    { icon: RotateCcw, label: '恢复默认', onClick: () => { if(confirm('确定?')) { resetSettings(); setCustomFontInput(''); } }, color: 'text-red-500 hover:bg-red-50 hover:border-red-200' },
                                    { icon: Download, label: '导出配置', onClick: handleExport, color: 'text-gray-600 hover:bg-gray-100' },
                                    { icon: Upload, label: '导入文件', action: 'file', color: 'text-gray-600 hover:bg-gray-100' },
                                    { icon: Type, label: '导入文本', onClick: () => setShowImportDialog(true), color: 'text-gray-600 hover:bg-gray-100' },
                                ].map((btn, i) => (
                                    btn.action === 'file' ? (
                                        <label key={i} className={`flex flex-col items-center justify-center gap-1.5 p-3 rounded-[16px] bg-white border border-gray-100 shadow-sm transition-colors cursor-pointer ${btn.color}`}>
                                            <btn.icon className="w-4 h-4 opacity-80" /> <span className="text-[11px] font-bold">{btn.label}</span>
                                            <input type="file" accept=".json" onChange={handleFileImport} className="hidden" />
                                        </label>
                                    ) : (
                                        <button key={i} onClick={btn.onClick} className={`flex flex-col items-center justify-center gap-1.5 p-3 rounded-[16px] bg-white border border-gray-100 shadow-sm transition-colors ${btn.color}`}>
                                            <btn.icon className="w-4 h-4 opacity-80" /> <span className="text-[11px] font-bold">{btn.label}</span>
                                        </button>
                                    )
                                ))}
                            </div>
                        </div>

                        {/* Floating Bottom Action */}
                        <div className="absolute inset-x-0 bottom-0 p-5 bg-gradient-to-t from-[#fbfbfc] via-[#fbfbfc]/90 to-transparent flex justify-center pointer-events-none">
                            <button
                                onClick={onClose}
                                className="pointer-events-auto bg-[#8871e4] hover:bg-[#725ec3] text-white font-bold text-[14px] py-3.5 px-8 flex items-center justify-center gap-2 rounded-[24px] shadow-[0_8px_20px_-6px_rgba(136,113,228,0.5)] transition-all transform hover:-translate-y-0.5 active:translate-y-0 active:shadow-sm"
                            >
                                <CheckCircle2 className="w-5 h-5" />
                                保存并关闭
                            </button>
                        </div>
                    </motion.div>

                    {/* 文本导入附赠弹窗保持基本干净即可 */}
                    {showImportDialog && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="fixed inset-0 bg-black/40 backdrop-blur-sm z-[60] flex items-center justify-center p-4" onClick={() => setShowImportDialog(false)}>
                            <div className="bg-white rounded-[24px] p-6 max-w-sm w-full shadow-2xl" onClick={(e) => e.stopPropagation()}>
                                <h3 className="text-[15px] font-bold mb-3">粘贴 JSON 配置</h3>
                                <textarea value={importText} onChange={(e) => setImportText(e.target.value)} className="w-full h-32 p-3 bg-gray-50 border border-gray-200 rounded-[16px] text-xs font-mono outline-none focus:ring-2 focus:ring-[#8871e4]/20 resize-none mb-4" />
                                <div className="flex gap-2">
                                    <button onClick={() => setShowImportDialog(false)} className="flex-1 py-2 bg-gray-100 text-gray-700 rounded-xl text-sm font-bold">取消</button>
                                    <button onClick={handleImport} className="flex-1 py-2 bg-[#8871e4] text-white rounded-xl text-sm font-bold">导入</button>
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
const ToggleSwitch = ({ checked, onChange, color = '#8871e4' }) => (
    <button onClick={() => onChange(!checked)} className="relative w-[42px] h-[24px] rounded-full transition-colors duration-200 outline-none flex-shrink-0" style={{ backgroundColor: checked ? color : '#e5e7eb' }}>
        <div className={`absolute top-[2px] left-[2px] w-[20px] h-[20px] bg-white rounded-full shadow-sm transition-transform duration-200 ${checked ? 'translate-x-[18px]' : ''}`} />
    </button>
);

/**
 * 三态开关：null=自动（跟随后端 config）/ true=强制开 / false=强制关
 * 用于检索增强调优面板，让用户无需 restart 后端即可会话级切换 feature flag。
 */
const TriStateToggle = ({ title, desc, value, onChange }) => {
    const options = [
        { v: null, label: '自动' },
        { v: true, label: '开' },
        { v: false, label: '关' },
    ];
    return (
        <div className="flex items-start justify-between gap-4 py-2">
            <div className="min-w-0 flex-1">
                <div className="text-[13px] font-bold text-gray-800">{title}</div>
                {desc && <div className="text-[11px] text-gray-500 mt-0.5 leading-relaxed">{desc}</div>}
            </div>
            <div className="flex items-center gap-1 bg-gray-100/80 p-0.5 rounded-[12px] flex-shrink-0">
                {options.map((opt) => {
                    const active = value === opt.v;
                    return (
                        <button
                            key={String(opt.v)}
                            onClick={() => onChange(opt.v)}
                            className={`px-2.5 py-1 rounded-[10px] text-[11px] font-bold transition-all ${
                                active
                                    ? 'bg-white text-violet-600 shadow-sm'
                                    : 'text-gray-500 hover:text-gray-700'
                            }`}
                        >
                            {opt.label}
                        </button>
                    );
                })}
            </div>
        </div>
    );
};

const KeyIcon = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4"/><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/></svg>
);

export default GlobalSettings;
