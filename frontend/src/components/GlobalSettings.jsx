import React, { useState } from 'react';
import { X, Type, ZoomIn, RotateCcw, Download, Upload, Check } from 'lucide-react';
import { useGlobalSettings, PRESET_FONTS } from '../contexts/GlobalSettingsContext';
import { motion, AnimatePresence } from 'framer-motion';

const GlobalSettings = ({ isOpen, onClose }) => {
    const {
        fontFamily,
        customFont,
        globalScale,
        setFontFamily,
        setCustomFont,
        setGlobalScale,
        resetSettings,
        exportSettings,
        importSettings,
        getCurrentFontName,
    } = useGlobalSettings();

    const [customFontInput, setCustomFontInput] = useState(customFont);
    const [showImportDialog, setShowImportDialog] = useState(false);
    const [importText, setImportText] = useState('');

    // 快捷缩放按钮
    const scalePresets = [
        { label: '75%', value: 0.75 },
        { label: '100%', value: 1.0 },
        { label: '125%', value: 1.25 },
        { label: '150%', value: 1.5 },
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
        const success = importSettings(importText);
        if (success) {
            alert('✅ 设置导入成功！');
            setShowImportDialog(false);
            setImportText('');
        } else {
            alert('❌ 导入失败，请检查 JSON 格式是否正确');
        }
    };

    const handleFileImport = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const success = importSettings(event.target.result);
                if (success) {
                    alert('✅ 设置导入成功！');
                } else {
                    alert('❌ 导入失败，请检查文件格式');
                }
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
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                onClick={onClose}
            >
                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                    transition={{ type: 'spring', damping: 20 }}
                    className="soft-panel rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-auto"
                    onClick={(e) => e.stopPropagation()}
                >
                    {/* Header */}
                    <div className="sticky top-0 bg-white/90 backdrop-blur-md border-b border-gray-100 p-6 flex items-center justify-between z-10">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                                <Type className="w-5 h-5 text-white" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold">全局设置</h2>
                                <p className="text-sm text-gray-500">自定义应用外观</p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-black/5 rounded-full transition-colors"
                        >
                            <X className="w-6 h-6" />
                        </button>
                    </div>

                    <div className="p-6 space-y-6">
                        {/* 字体设置 */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2">
                                <Type className="w-5 h-5 text-blue-600" />
                                <h3 className="text-lg font-semibold">字体设置</h3>
                            </div>

                            {/* 预设字体选择 */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-700">预设字体</label>
                                <div className="grid grid-cols-2 gap-2">
                                    {PRESET_FONTS.map((font) => (
                                        <button
                                            key={font.id}
                                            onClick={() => setFontFamily(font.id)}
                                            className={`p-3 rounded-xl transition-all relative overflow-hidden group ${fontFamily === font.id
                                                ? 'soft-card ring-2 ring-blue-500 bg-blue-50/50'
                                                : 'soft-card hover:bg-[var(--color-bg-subtle)]'
                                                }`}
                                            style={{ fontFamily: font.value }}
                                        >
                                            <div className="flex items-center justify-between relative z-10">
                                                <span className="font-medium text-gray-800">{font.name}</span>
                                                {fontFamily === font.id && (
                                                    <div className="bg-blue-500 rounded-full p-0.5">
                                                        <Check className="w-3 h-3 text-white" />
                                                    </div>
                                                )}
                                            </div>
                                            <div className="text-xs text-gray-500 mt-1 relative z-10">ABC 中文 123</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* 自定义字体 */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-700">
                                    自定义字体 <span className="text-gray-400">(Google Fonts)</span>
                                </label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={customFontInput}
                                        onChange={(e) => setCustomFontInput(e.target.value)}
                                        placeholder="例如: Noto Serif SC"
                                        className="flex-1 px-4 py-3 soft-input rounded-xl outline-none text-sm"
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter') applyCustomFont();
                                        }}
                                    />
                                    <button
                                        onClick={applyCustomFont}
                                        className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-colors font-medium"
                                    >
                                        应用
                                    </button>
                                </div>
                                <p className="text-xs text-gray-500">
                                    输入任意 Google Fonts 字体名称，例如: Noto Serif SC, Playfair Display
                                </p>
                            </div>

                            {/* 字体预览 */}
                            <div className="p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl">
                                <div className="text-xs text-gray-500 mb-2">当前字体预览：{getCurrentFontName()}</div>
                                <div className="text-2xl font-medium mb-2">
                                    The quick brown fox jumps over the lazy dog
                                </div>
                                <div className="text-xl">
                                    你好世界！ChatPDF 全局设置功能测试 0123456789
                                </div>
                            </div>
                        </div>

                        {/* 分割线 */}
                        <div className="border-t border-gray-200"></div>

                        {/* 字体大小设置 */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2">
                                <ZoomIn className="w-5 h-5 text-purple-600" />
                                <h3 className="text-lg font-semibold">字体大小</h3>
                            </div>

                            {/* 当前字体大小显示 */}
                            <div className="text-center">
                                <div className="text-4xl font-bold text-blue-600">
                                    {Math.round(16 * globalScale)}px
                                </div>
                                <div className="text-sm text-gray-500 mt-1">当前基准字体大小</div>
                            </div>

                            {/* 字体大小滑块 */}
                            <div className="space-y-2">
                                <input
                                    type="range"
                                    min="50"
                                    max="200"
                                    step="5"
                                    value={globalScale * 100}
                                    onChange={(e) => setGlobalScale(parseInt(e.target.value) / 100)}
                                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                                    style={{
                                        background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${((globalScale - 0.5) / 1.5) * 100}%, #E5E7EB ${((globalScale - 0.5) / 1.5) * 100}%, #E5E7EB 100%)`
                                    }}
                                />
                                <div className="flex justify-between text-xs text-gray-500">
                                    <span>8px</span>
                                    <span>16px</span>
                                    <span>32px</span>
                                </div>
                            </div>

                            {/* 快捷字体大小按钮 */}
                            <div className="grid grid-cols-4 gap-2">
                                {scalePresets.map((preset) => (
                                    <button
                                        key={preset.value}
                                        onClick={() => setGlobalScale(preset.value)}
                                        className={`py-2 rounded-lg transition-all font-medium ${Math.abs(globalScale - preset.value) < 0.01
                                            ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-lg shadow-blue-500/30 transform scale-105'
                                            : 'soft-card text-gray-600 hover:text-gray-900'
                                            }`}
                                    >
                                        {Math.round(16 * preset.value)}px
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* 分割线 */}
                        <div className="border-t border-gray-200"></div>

                        {/* 操作按钮 */}
                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-gray-700">操作</h3>

                            <div className="grid grid-cols-2 gap-3">
                                {/* 重置设置 */}
                                <button
                                    onClick={() => {
                                        if (confirm('确定要重置所有设置为默认值吗？')) {
                                            resetSettings();
                                            setCustomFontInput('');
                                        }
                                    }}
                                    className="soft-card flex items-center justify-center gap-2 px-4 py-3 text-gray-700 hover:text-red-600 hover:bg-red-50/50 rounded-xl transition-all"
                                >
                                    <RotateCcw className="w-4 h-4" />
                                    <span className="font-medium">重置设置</span>
                                </button>

                                {/* 导出设置 */}
                                <button
                                    onClick={handleExport}
                                    className="soft-card flex items-center justify-center gap-2 px-4 py-3 text-green-700 bg-green-50/30 hover:bg-green-50/80 rounded-xl transition-all"
                                >
                                    <Download className="w-4 h-4" />
                                    <span className="font-medium">导出设置</span>
                                </button>

                                {/* 导入设置（文件） */}
                                <label className="soft-card flex items-center justify-center gap-2 px-4 py-3 text-blue-700 bg-blue-50/30 hover:bg-blue-50/80 rounded-xl transition-all cursor-pointer col-span-1">
                                    <Upload className="w-4 h-4" />
                                    <span className="font-medium">导入文件</span>
                                    <input
                                        type="file"
                                        accept=".json"
                                        onChange={handleFileImport}
                                        className="hidden"
                                    />
                                </label>

                                {/* 导入设置（文本） */}
                                <button
                                    onClick={() => setShowImportDialog(true)}
                                    className="soft-card flex items-center justify-center gap-2 px-4 py-3 text-purple-700 bg-purple-50/30 hover:bg-purple-50/80 rounded-xl transition-all"
                                >
                                    <Upload className="w-4 h-4" />
                                    <span className="font-medium">粘贴文本</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* 导入文本对话框 */}
                {showImportDialog && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[60] flex items-center justify-center p-4"
                        onClick={() => setShowImportDialog(false)}
                    >
                        <div
                            className="soft-panel rounded-2xl p-6 max-w-md w-full"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <h3 className="text-xl font-bold mb-4">导入设置 (JSON)</h3>
                            <textarea
                                value={importText}
                                onChange={(e) => setImportText(e.target.value)}
                                placeholder='粘贴设置 JSON 文本...'
                                className="w-full h-48 px-4 py-3 soft-input rounded-xl outline-none text-sm font-mono resize-none"
                            />
                            <div className="flex gap-3 mt-4">
                                <button
                                    onClick={handleImport}
                                    className="flex-1 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-colors font-medium"
                                >
                                    导入
                                </button>
                                <button
                                    onClick={() => {
                                        setShowImportDialog(false);
                                        setImportText('');
                                    }}
                                    className="flex-1 py-3 bg-gray-200 hover:bg-gray-300 rounded-xl transition-colors font-medium"
                                >
                                    取消
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </motion.div>
        </AnimatePresence>
    );
};

export default GlobalSettings;
