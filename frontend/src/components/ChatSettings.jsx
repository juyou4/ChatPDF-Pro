import React, { useState, useEffect } from 'react';
import { X, SlidersHorizontal, HelpCircle, RotateCcw, Plus, Trash2 } from 'lucide-react';
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * 对话设置面板
 * 参考 cherry-studio 风格，包含：模型温度、Top-P、上下文数、最大 Token 数、流式输出、自定义参数
 * 每个参数行：标签 + 问号提示 | 开关（右侧）
 * 开关启用时显示：滑块（80%宽度）+ 数字输入框（20%宽度）
 */
const ChatSettings = ({ isOpen, onClose }) => {
    const {
        maxTokens,
        temperature,
        topP,
        contextCount,
        streamOutput,
        enableTemperature,
        enableTopP,
        enableMaxTokens,
        customParams,
        setMaxTokens,
        setTemperature,
        setTopP,
        setContextCount,
        setStreamOutput,
        setEnableTemperature,
        setEnableTopP,
        setEnableMaxTokens,
        setCustomParams,
        DEFAULT_SETTINGS,
    } = useGlobalSettings();

    // 重置对话参数
    const resetChatSettings = () => {
        setMaxTokens(DEFAULT_SETTINGS.maxTokens);
        setTemperature(DEFAULT_SETTINGS.temperature);
        setTopP(DEFAULT_SETTINGS.topP);
        setContextCount(DEFAULT_SETTINGS.contextCount);
        setStreamOutput(DEFAULT_SETTINGS.streamOutput);
        setEnableTemperature(DEFAULT_SETTINGS.enableTemperature);
        setEnableTopP(DEFAULT_SETTINGS.enableTopP);
        setEnableMaxTokens(DEFAULT_SETTINGS.enableMaxTokens);
        setCustomParams(DEFAULT_SETTINGS.customParams);
    };

    // 添加自定义参数
    const addCustomParam = () => {
        setCustomParams([...customParams, { name: '', type: 'string', value: '' }]);
    };

    // 更新自定义参数
    const updateCustomParam = (index, field, val) => {
        const updated = [...customParams];
        if (field === 'type') {
            // 切换类型时重置值
            updated[index] = { ...updated[index], type: val, value: val === 'boolean' ? false : val === 'number' ? 0 : '' };
        } else {
            updated[index] = { ...updated[index], [field]: val };
        }
        setCustomParams(updated);
    };

    // 删除自定义参数
    const removeCustomParam = (index) => {
        setCustomParams(customParams.filter((_, i) => i !== index));
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

                        {/* 模型温度 — 带开关 + 滑块 + 数字输入框 */}
                        <SettingToggleSlider
                            label="模型温度"
                            tooltip="控制回答的随机性。值越低越精确，值越高越有创造性"
                            enabled={enableTemperature}
                            onToggle={setEnableTemperature}
                            value={temperature}
                            onChange={setTemperature}
                            min={0} max={2} step={0.1}
                            precision={1}
                            color="emerald"
                        />

                        <div className="border-t border-gray-100"></div>

                        {/* Top-P — 带开关 + 滑块 + 数字输入框 */}
                        <SettingToggleSlider
                            label="Top-P"
                            tooltip="核采样参数。控制候选词的概率范围，值越小回答越集中"
                            enabled={enableTopP}
                            onToggle={setEnableTopP}
                            value={topP}
                            onChange={setTopP}
                            min={0} max={1} step={0.05}
                            precision={2}
                            color="emerald"
                        />

                        <div className="border-t border-gray-100"></div>

                        {/* 上下文数 — 无开关，滑块 + 数字输入框 */}
                        <SettingSliderWithInput
                            label="上下文数"
                            tooltip="发送给模型的历史消息轮数。值越大模型记忆越多，但消耗更多 Token"
                            value={contextCount}
                            onChange={(v) => setContextCount(Math.round(v))}
                            min={0} max={50} step={1}
                            precision={0}
                            color="emerald"
                        />

                        <div className="border-t border-gray-100"></div>

                        {/* 最大 Token 数 — 带开关 + 滑块 + 数字输入框 */}
                        <SettingToggleSlider
                            label="最大 Token 数"
                            tooltip="限制模型单次回复的最大长度。关闭则由模型自行决定"
                            enabled={enableMaxTokens}
                            onToggle={setEnableMaxTokens}
                            value={maxTokens}
                            onChange={setMaxTokens}
                            min={512} max={32768} step={512}
                            precision={0}
                            color="emerald"
                        />

                        <div className="border-t border-gray-100"></div>

                        {/* 流式输出 — 保持不变 */}
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-semibold text-gray-800">流式输出</span>
                                <Tooltip text="开启后回答会逐字显示，关闭则等待完整回答后一次性显示" />
                            </div>
                            <ToggleSwitch checked={streamOutput} onChange={setStreamOutput} />
                        </div>

                        <div className="border-t border-gray-100"></div>

                        {/* 自定义参数区域 */}
                        <div className="space-y-3">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">自定义参数</span>
                                    <Tooltip text="添加任意 key-value 参数直接传给 API，如 DeepSeek 的 enable_search" />
                                </div>
                                <button
                                    onClick={addCustomParam}
                                    className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium text-emerald-600 bg-emerald-50 hover:bg-emerald-100 rounded-lg transition-colors"
                                >
                                    <Plus className="w-3.5 h-3.5" />
                                    添加参数
                                </button>
                            </div>

                            {/* 自定义参数列表 */}
                            {customParams.length > 0 && (
                                <div className="space-y-2">
                                    {customParams.map((param, index) => (
                                        <CustomParamRow
                                            key={index}
                                            param={param}
                                            onChange={(field, val) => updateCustomParam(index, field, val)}
                                            onRemove={() => removeCustomParam(index)}
                                        />
                                    ))}
                                </div>
                            )}

                            {customParams.length === 0 && (
                                <p className="text-xs text-gray-400 text-center py-2">暂无自定义参数</p>
                            )}
                        </div>

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

/**
 * 数字输入框组件
 * 输入时实时更新本地 state，失焦时 clamp 并提交
 */
const NumberInput = ({ value, onChange, min, max, step, precision, disabled }) => {
    const [localValue, setLocalValue] = useState(String(value));

    // 外部值变化时同步本地 state
    useEffect(() => {
        const formatted = precision > 0 ? Number(value).toFixed(precision) : String(Math.round(value));
        setLocalValue(formatted);
    }, [value, precision]);

    const handleBlur = () => {
        let num = parseFloat(localValue);
        if (isNaN(num)) {
            num = value; // 无效输入恢复原值
        }
        // clamp 到 min/max 范围
        num = Math.min(Math.max(num, min), max);
        // 按精度格式化
        const formatted = precision > 0 ? Number(num).toFixed(precision) : String(Math.round(num));
        setLocalValue(formatted);
        onChange(Number(formatted));
    };

    return (
        <input
            type="text"
            value={localValue}
            onChange={(e) => setLocalValue(e.target.value)}
            onBlur={handleBlur}
            disabled={disabled}
            className={`w-full text-center text-sm font-mono border rounded-lg px-2 py-1.5 outline-none transition-colors
                ${disabled
                    ? 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
                    : 'bg-white text-gray-700 border-gray-200 focus:border-emerald-400 focus:ring-1 focus:ring-emerald-200'
                }`}
        />
    );
};

/**
 * 带开关的滑块 + 数字输入框设置行
 * 布局：标签 + 提示 | 开关
 * 启用时：滑块（80%）+ 数字输入框（20%）
 * 禁用时：滑块和输入框变灰色禁用状态
 */
const SettingToggleSlider = ({ label, tooltip, enabled, onToggle, value, onChange, min, max, step, precision, color }) => {
    const pct = ((value - min) / (max - min)) * 100;
    const c = color || 'emerald';
    const gradientColors = { emerald: '#10B981', blue: '#3B82F6', orange: '#F59E0B' };
    const gc = gradientColors[c] || gradientColors.emerald;

    return (
        <div className="space-y-3">
            {/* 标题行：标签 + 提示 | 开关 */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-800">{label}</span>
                    {tooltip && <Tooltip text={tooltip} />}
                </div>
                <ToggleSwitch checked={enabled} onChange={onToggle} />
            </div>
            {/* 滑块（80%）+ 数字输入框（20%）— 始终显示，禁用时灰色 */}
            <div className="flex items-center gap-3">
                <div className="flex-[4]">
                    <input
                        type="range"
                        min={min} max={max} step={step}
                        value={value}
                        onChange={(e) => onChange(parseFloat(e.target.value))}
                        disabled={!enabled}
                        className={`w-full h-1.5 rounded-lg appearance-none ${enabled ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
                        style={{
                            background: enabled
                                ? `linear-gradient(to right, ${gc} 0%, ${gc} ${pct}%, #E5E7EB ${pct}%, #E5E7EB 100%)`
                                : '#E5E7EB'
                        }}
                    />
                </div>
                <div className="flex-[1]">
                    <NumberInput
                        value={value}
                        onChange={onChange}
                        min={min} max={max} step={step}
                        precision={precision}
                        disabled={!enabled}
                    />
                </div>
            </div>
        </div>
    );
};

/**
 * 滑块 + 数字输入框设置行（无开关）
 * 用于上下文数等始终启用的参数
 */
const SettingSliderWithInput = ({ label, tooltip, value, onChange, min, max, step, precision, color }) => {
    const pct = ((value - min) / (max - min)) * 100;
    const c = color || 'emerald';
    const gradientColors = { emerald: '#10B981', blue: '#3B82F6', orange: '#F59E0B' };
    const gc = gradientColors[c] || gradientColors.emerald;

    return (
        <div className="space-y-3">
            {/* 标题行 */}
            <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-gray-800">{label}</span>
                {tooltip && <Tooltip text={tooltip} />}
            </div>
            {/* 滑块（80%）+ 数字输入框（20%） */}
            <div className="flex items-center gap-3">
                <div className="flex-[4]">
                    <input
                        type="range"
                        min={min} max={max} step={step}
                        value={value}
                        onChange={(e) => onChange(parseFloat(e.target.value))}
                        className="w-full h-1.5 rounded-lg appearance-none cursor-pointer"
                        style={{ background: `linear-gradient(to right, ${gc} 0%, ${gc} ${pct}%, #E5E7EB ${pct}%, #E5E7EB 100%)` }}
                    />
                </div>
                <div className="flex-[1]">
                    <NumberInput
                        value={value}
                        onChange={onChange}
                        min={min} max={max} step={step}
                        precision={precision}
                        disabled={false}
                    />
                </div>
            </div>
        </div>
    );
};

/**
 * 自定义参数行
 * 包含：参数名输入框、类型选择（string/number/boolean）、值输入框、删除按钮
 */
const CustomParamRow = ({ param, onChange, onRemove }) => {
    return (
        <div className="flex items-center gap-2 p-2 bg-gray-50 rounded-lg">
            {/* 参数名 */}
            <input
                type="text"
                value={param.name}
                onChange={(e) => onChange('name', e.target.value)}
                placeholder="参数名"
                className="flex-[2] text-sm border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:border-emerald-400 focus:ring-1 focus:ring-emerald-200 bg-white"
            />
            {/* 类型选择 */}
            <select
                value={param.type}
                onChange={(e) => onChange('type', e.target.value)}
                className="flex-[1] text-sm border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:border-emerald-400 focus:ring-1 focus:ring-emerald-200 bg-white cursor-pointer"
            >
                <option value="string">string</option>
                <option value="number">number</option>
                <option value="boolean">boolean</option>
            </select>
            {/* 值输入 — 根据类型不同渲染不同控件 */}
            {param.type === 'boolean' ? (
                <div className="flex-[2] flex justify-center">
                    <ToggleSwitch
                        checked={!!param.value}
                        onChange={(v) => onChange('value', v)}
                    />
                </div>
            ) : (
                <input
                    type={param.type === 'number' ? 'number' : 'text'}
                    value={param.value}
                    onChange={(e) => onChange('value', param.type === 'number' ? Number(e.target.value) : e.target.value)}
                    placeholder="值"
                    className="flex-[2] text-sm border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:border-emerald-400 focus:ring-1 focus:ring-emerald-200 bg-white"
                />
            )}
            {/* 删除按钮 */}
            <button
                onClick={onRemove}
                className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
            >
                <Trash2 className="w-4 h-4" />
            </button>
        </div>
    );
};

export default ChatSettings;
