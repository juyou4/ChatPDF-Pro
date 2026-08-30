import React, { useState } from 'react';
import { ChevronLeft, Type, RotateCcw, Download, Upload, Brain, CheckCircle2, Library } from 'lucide-react';
import MemoryPanel from './MemoryPanel';
import PaperLibraryPanel from './PaperLibraryPanel';
import { useFontSettings } from '../contexts/FontSettingsContext';
import { useChatParams } from '../contexts/ChatParamsContext';
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';
import { useWebSearch } from '../contexts/WebSearchContext';
import { useReadingSettings } from '../contexts/ReadingSettingsContext';
import { motion, AnimatePresence } from 'framer-motion';
import SettingsSegmentedControl from './SettingsSegmentedControl';
import SettingsRange from './SettingsRange';

const GlobalSettings = ({ isOpen, onClose }) => {
    // 字体选择已上移到设置中心一级「界面」tab；此处仅保留重置能力
    const { resetFontSettings } = useFontSettings();
    // 检索增强调优已上移到设置中心一级「检索」tab；此处仅保留重置能力
    const {
        enableMemory, setEnableMemory,
        memoryTopK, setMemoryTopK,
        memoryInjectionBudget, setMemoryInjectionBudget,
        memoryPrivacyMode, setMemoryPrivacyMode,
        resetChatParams,
    } = useChatParams();
    const [activeSection, setActiveSection] = useState('services');
    const { exportSettings, importSettings } = useGlobalSettings();
    // 阅读相关设置已上移到设置中心一级「阅读」tab；此处仅保留重置能力
    const { resetReadingSettings } = useReadingSettings();

    const resetSettings = () => { resetFontSettings(); resetChatParams(); resetWebSearch(); resetReadingSettings(); };

    const [showMemoryPanel, setShowMemoryPanel] = useState(false);
    const [showPaperLibraryPanel, setShowPaperLibraryPanel] = useState(false);
    const [showImportDialog, setShowImportDialog] = useState(false);
    const [importText, setImportText] = useState('');

    // 联网搜索已上移到设置中心一级「检索」tab；此处仅保留重置能力
    const { resetWebSearch } = useWebSearch();

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

                        {/* 「阅读」「字体」「联网搜索」均已上移到设置中心一级 tab，
                            避免同一设置出现在两处；这里只剩服务与高级两个分区 */}
                        <div className="settings-segment relative grid grid-cols-2 mx-7 mb-4 p-1 rounded-[20px] flex-shrink-0" role="tablist" aria-label="全局设置分类">
                            <span
                                aria-hidden="true"
                                className="settings-segment-indicator absolute left-1 top-1 bottom-1 rounded-[16px] transition-transform duration-[320ms] ease-[cubic-bezier(0.4,0,0.2,1)] motion-reduce:transition-none"
                                style={{
                                    width: 'calc((100% - 0.5rem) / 2)',
                                    transform: `translateX(${['services', 'advanced'].indexOf(activeSection) * 100}%)`,
                                }}
                            />
                            {[
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
                            {activeSection === 'services' && (
                            <>
                            {/* Memory Settings */}
                            <div className="settings-card bg-white p-5 border border-gray-200/90">
                                <div className="flex items-center justify-between gap-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-blue-50 text-blue-500 flex items-center justify-center">
                                            <Brain className="w-4 h-4" />
                                        </div>
                                        <div>
                                            <h3 className="text-[14px] font-bold text-gray-900">智能记忆系统</h3>
                                            <p className="text-[12px] text-gray-500">记住偏好和重要信息</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <button onClick={() => setShowMemoryPanel(true)} className="text-[12px] font-bold text-blue-600 bg-blue-50 hover:bg-blue-100 px-3 py-1.5 rounded-xl transition-colors">
                                            管理记忆
                                        </button>
                                        <ToggleSwitch checked={enableMemory} onChange={setEnableMemory} color="#3b82f6" />
                                    </div>
                                </div>
                                <div className="mt-4 pt-4 border-t border-gray-100 flex items-center justify-between gap-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-indigo-50 text-indigo-500 flex items-center justify-center">
                                            <Library className="w-4 h-4" />
                                        </div>
                                        <div>
                                            <h3 className="text-[14px] font-bold text-gray-900">论文订阅库</h3>
                                            <p className="text-[12px] text-gray-500">按订阅追踪新论文</p>
                                        </div>
                                    </div>
                                    <button onClick={() => setShowPaperLibraryPanel(true)} className="text-[12px] font-bold text-indigo-600 bg-indigo-50 hover:bg-indigo-100 px-3 py-1.5 rounded-xl transition-colors">
                                        打开论文库
                                    </button>
                                </div>
                                {enableMemory && (
                                    <div className="mt-4 space-y-3">
                                        <MemoryOverrideSlider
                                            label="记忆检索条数"
                                            desc="每轮最多带几条记忆"
                                            value={memoryTopK}
                                            onChange={setMemoryTopK}
                                            min={1}
                                            max={20}
                                            step={1}
                                            fallbackValue={3}
                                        />
                                        <MemoryOverrideSlider
                                            label="记忆注入预算"
                                            desc="记忆最多占用多少 token"
                                            value={memoryInjectionBudget}
                                            onChange={setMemoryInjectionBudget}
                                            min={100}
                                            max={3000}
                                            step={100}
                                            fallbackValue={800}
                                        />
                                        <div className="bg-gray-50/80 p-3 rounded-[16px] border border-gray-100/80">
                                            <div className="flex items-start justify-between gap-3">
                                                <div className="min-w-0 flex-1">
                                                    <div className="text-[13px] font-bold text-gray-800">共享 / 演示模式</div>
                                                    <div className="text-[11px] text-gray-500 mt-0.5 leading-snug">
                                                        只保留当前文档事实，适合录屏或分享
                                                    </div>
                                                </div>
                                                <ToggleSwitch
                                                    checked={memoryPrivacyMode === 'shared'}
                                                    onChange={(next) => setMemoryPrivacyMode(next ? 'shared' : 'personal')}
                                                    color="#3b82f6"
                                                />
                                            </div>
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
                                    { icon: RotateCcw, label: '恢复默认', onClick: () => { if(confirm('确定?')) resetSettings(); }, color: 'text-red-500 hover:bg-red-50 hover:border-red-200' },
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
                    <PaperLibraryPanel isOpen={showPaperLibraryPanel} onClose={() => setShowPaperLibraryPanel(false)} />
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

/**
 * 记忆参数滑块：null 表示"跟随后端 config"，拖动即转为会话级覆盖。
 * 和检索增强的 tri-state 是同一套语义，只是这里是数值而非开关。
 */
const MemoryOverrideSlider = ({ label, desc, value, onChange, min, max, step, fallbackValue }) => {
    const isOverridden = value !== null && value !== undefined;
    const displayValue = isOverridden ? value : fallbackValue;
    return (
        <div className="bg-gray-50/80 p-3 rounded-[16px] border border-gray-100/80">
            <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                    <div className="text-[13px] font-bold text-gray-800">{label}</div>
                    {desc && <div className="text-[11px] text-gray-500 mt-0.5 leading-relaxed">{desc}</div>}
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                    <span className={`text-[12px] font-bold tabular-nums ${isOverridden ? 'text-blue-600' : 'text-gray-400'}`}>
                        {displayValue}
                    </span>
                    {isOverridden && (
                        <button
                            onClick={() => onChange(null)}
                            className="text-[11px] font-bold text-gray-500 hover:text-gray-700 bg-white border border-gray-200 px-2 py-0.5 rounded-[9px] transition-colors"
                        >
                            跟随后端
                        </button>
                    )}
                </div>
            </div>
            <SettingsRange
                ariaLabel={label}
                min={String(min)}
                max={String(max)}
                step={String(step)}
                value={displayValue}
                onChange={(next) => onChange(Math.round(next))}
                className="mt-3"
            />
            {!isOverridden && (
                <div className="text-[11px] text-gray-400 mt-1.5">当前跟随后端配置，拖动滑块即可本地覆盖</div>
            )}
        </div>
    );
};

const KeyIcon = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4"/><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/></svg>
);

export default GlobalSettings;
