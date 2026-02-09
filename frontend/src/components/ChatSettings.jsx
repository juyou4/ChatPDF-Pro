import React from 'react';
import { X, SlidersHorizontal, HelpCircle, RotateCcw } from 'lucide-react';
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * 对话设置面板
 * 参考 cherry-studio 风格，包含：模型温度、Top-P、上下文数、最大 Token 数、流式输出
 */
const ChatSettings = ({ isOpen, onClose }) => {
    const {
        maxTokens,
        temperature,
        topP,
        contextCount,
        streamOutput,
        setMaxTokens,
        setTemperature,
        setTopP,
        setContextCount,
        setStreamOutput,
        DEFAULT_SETTINGS,
    } = useGlobalSettings();

    // 重置对话参数
    const resetChatSettings = () => {
        setMaxTokens(DEFAULT_SETTINGS.maxTokens);
        setTemperature(DEFAULT_SETTINGS.temperature);
        setTopP(DEFAULT_SETTINGS.topP);
        setContextCount(DEFAULT_SETTINGS.contextCount);
        setStreamOutput(DEFAULT_SETTINGS.streamOutput);
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
                    className="soft-panel rounded-2xl shadow-2xl max-w-lg w-full max-h-[90vh] overflow-auto"
                    onClick={(e) => e.stopPropagation()}
                >
                    {/* 头部 */}
                    <div className="sticky top-0 bg-white/90 backdrop-blur-md border-b border-gray-100 p-6 flex items-center justify-between z-10">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center">
                                <SlidersHorizontal className="w-5 h-5 text-white" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold">对话设置</h2>
                                <p className="text-sm text-gray-500">调整模型生成参数</p>
                            </div>
                        </div>
                        <button onClick={onClose} className="p-2 hover:bg-black/5 rounded-full transition-colors">
                            <X className="w-6 h-6" />
                        </button>
                    </div>

                    <div className="p-6 space-y-6">

                        {/* 模型温度 */}
                        <SettingSlider
                            label="模型温度"
                            tooltip="控制回答的随机性。值越低越精确，值越高越有创造性"
                            value={temperature}
                            onChange={setTemperature}
                            min={0} max={2} step={0.1}
                            displayValue={temperature.toFixed(1)}
                            color="emerald"
                            marks={['0', '0.7', '2']}
                        />

                        {/* 分割线 */}
                        <div className="border-t border-gray-100"></div>

                        {/* Top-P */}
                        <SettingSlider
                            label="Top-P"
                            tooltip="核采样参数。控制候选词的概率范围，值越小回答越集中"
                            value={topP}
                            onChange={setTopP}
                            min={0} max={1} step={0.05}
                            displayValue={topP.toFixed(2)}
                            color="emerald"
                            marks={['0', '1']}
                        />

                        {/* 分割线 */}
                        <div className="border-t border-gray-100"></div>

                        {/* 上下文数 */}
                        <SettingSlider
                            label="上下文数"
                            tooltip="发送给模型的历史消息轮数。值越大模型记忆越多，但消耗更多 Token"
                            value={contextCount}
                            onChange={(v) => setContextCount(Math.round(v))}
                            min={0} max={50} step={1}
                            displayValue={contextCount === 50 ? '不限' : String(contextCount)}
                            color="emerald"
                            marks={['0', '25', '50']}
                        />

                        {/* 分割线 */}
                        <div className="border-t border-gray-100"></div>

                        {/* 最大 Token 数 */}
                        <SettingToggleSlider
                            label="最大 Token 数"
                            tooltip="限制模型单次回复的最大长度。关闭则由模型自行决定"
                            enabled={maxTokens > 0}
                            onToggle={(on) => setMaxTokens(on ? DEFAULT_SETTINGS.maxTokens : 0)}
                            value={maxTokens}
                            onChange={setMaxTokens}
                            min={512} max={32768} step={512}
                            displayValue={maxTokens > 0 ? String(maxTokens) : '自动'}
                            color="emerald"
                            marks={['512', '8192', '32768']}
                        />

                        {/* 分割线 */}
                        <div className="border-t border-gray-100"></div>

                        {/* 流式输出 */}
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-semibold text-gray-800">流式输出</span>
                                <Tooltip text="开启后回答会逐字显示，关闭则等待完整回答后一次性显示" />
                            </div>
                            <ToggleSwitch checked={streamOutput} onChange={setStreamOutput} />
                        </div>

                        {/* 分割线 */}
                        <div className="border-t border-gray-100"></div>

                        {/* 重置按钮 */}
                        <button
                            onClick={() => { if (confirm('确定要重置所有对话参数为默认值吗？')) resetChatSettings(); }}
                            className="soft-card w-full flex items-center justify-center gap-2 px-4 py-3 text-gray-700 hover:text-red-600 hover:bg-red-50/50 rounded-xl transition-all"
                        >
                            <RotateCcw className="w-4 h-4" />
                            <span className="font-medium">重置为默认值</span>
                        </button>
                    </div>
                </motion.div>
            </motion.div>
        </AnimatePresence>
    );
};

/* ========== 子组件 ========== */

/** 问号提示气泡 */
const Tooltip = ({ text }) => (
    <div className="group relative">
        <HelpCircle className="w-3.5 h-3.5 text-gray-400 cursor-help" />
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50 max-w-[240px] whitespace-normal text-center">
            {text}
        </div>
    </div>
);

/** 开关组件 — 仿 cherry-studio 绿色胶囊 */
const ToggleSwitch = ({ checked, onChange }) => (
    <button
        onClick={() => onChange(!checked)}
        className={`relative w-11 h-6 rounded-full transition-colors ${checked ? 'bg-emerald-500' : 'bg-gray-300'}`}
    >
        <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${checked ? 'translate-x-5' : ''}`} />
    </button>
);

/** 滑块设置行 */
const SettingSlider = ({ label, tooltip, value, onChange, min, max, step, displayValue, color, marks }) => {
    const pct = ((value - min) / (max - min)) * 100;
    const c = color || 'emerald';
    const gradientColors = { emerald: '#10B981', blue: '#3B82F6', orange: '#F59E0B' };
    const gc = gradientColors[c] || gradientColors.emerald;

    return (
        <div className="space-y-3">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-800">{label}</span>
                    {tooltip && <Tooltip text={tooltip} />}
                </div>
                <span className="text-sm font-mono text-gray-600 bg-gray-100 px-2 py-0.5 rounded">{displayValue}</span>
            </div>
            <input
                type="range"
                min={min} max={max} step={step}
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                className="w-full h-1.5 rounded-lg appearance-none cursor-pointer"
                style={{ background: `linear-gradient(to right, ${gc} 0%, ${gc} ${pct}%, #E5E7EB ${pct}%, #E5E7EB 100%)` }}
            />
            {marks && (
                <div className="flex justify-between text-xs text-gray-400">
                    {marks.map((m, i) => <span key={i}>{m}</span>)}
                </div>
            )}
        </div>
    );
};

/** 带开关的滑块设置行 */
const SettingToggleSlider = ({ label, tooltip, enabled, onToggle, value, onChange, min, max, step, displayValue, color, marks }) => {
    const pct = enabled ? ((value - min) / (max - min)) * 100 : 0;
    const c = color || 'emerald';
    const gradientColors = { emerald: '#10B981', blue: '#3B82F6', orange: '#F59E0B' };
    const gc = gradientColors[c] || gradientColors.emerald;

    return (
        <div className="space-y-3">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-800">{label}</span>
                    {tooltip && <Tooltip text={tooltip} />}
                </div>
                <div className="flex items-center gap-3">
                    {enabled && <span className="text-sm font-mono text-gray-600 bg-gray-100 px-2 py-0.5 rounded">{displayValue}</span>}
                    <ToggleSwitch checked={enabled} onChange={onToggle} />
                </div>
            </div>
            {enabled && (
                <>
                    <input
                        type="range"
                        min={min} max={max} step={step}
                        value={value}
                        onChange={(e) => onChange(parseFloat(e.target.value))}
                        className="w-full h-1.5 rounded-lg appearance-none cursor-pointer"
                        style={{ background: `linear-gradient(to right, ${gc} 0%, ${gc} ${pct}%, #E5E7EB ${pct}%, #E5E7EB 100%)` }}
                    />
                    {marks && (
                        <div className="flex justify-between text-xs text-gray-400">
                            {marks.map((m, i) => <span key={i}>{m}</span>)}
                        </div>
                    )}
                </>
            )}
        </div>
    );
};

export default ChatSettings;
