import React, { useState, useEffect } from 'react';
import { ChevronLeft, SlidersHorizontal, HelpCircle, RotateCcw, Plus, Trash2, Code, MessageSquare, Type, Sigma } from 'lucide-react';
import { useChatParams, CHAT_PARAMS_DEFAULT_SETTINGS } from '../contexts/ChatParamsContext';
import { motion, AnimatePresence } from 'framer-motion';
import SettingsSegmentedControl from './SettingsSegmentedControl';

/**
 * 对话设置面板
 * 采用全局极简圆润卡片 UI
 */
const ChatSettings = ({ isOpen, onClose }) => {
    const {
        maxTokens, temperature, topP, contextCount, streamOutput,
        enableTemperature, enableTopP, enableMaxTokens, customParams,
        thoughtAutoCollapse, answerDetailLevel, sendShortcut,
        confirmDeleteMessage, confirmRegenerateMessage,
        codeCollapsible, codeWrappable, codeShowLineNumbers,
        mathEngine, mathEnableSingleDollar, messageStyle, messageFontSize,
        setMaxTokens, setTemperature, setTopP, setContextCount, setStreamOutput,
        setEnableTemperature, setEnableTopP, setEnableMaxTokens, setCustomParams,
        setThoughtAutoCollapse, setAnswerDetailLevel, setSendShortcut,
        setConfirmDeleteMessage, setConfirmRegenerateMessage,
        setCodeCollapsible, setCodeWrappable, setCodeShowLineNumbers,
        setMathEngine, setMathEnableSingleDollar, setMessageStyle, setMessageFontSize,
    } = useChatParams();

    const DEFAULT_SETTINGS = CHAT_PARAMS_DEFAULT_SETTINGS;

    const resetChatSettings = () => {
        setMaxTokens(DEFAULT_SETTINGS.maxTokens); setTemperature(DEFAULT_SETTINGS.temperature);
        setTopP(DEFAULT_SETTINGS.topP); setContextCount(DEFAULT_SETTINGS.contextCount);
        setStreamOutput(DEFAULT_SETTINGS.streamOutput); setEnableTemperature(DEFAULT_SETTINGS.enableTemperature);
        setEnableTopP(DEFAULT_SETTINGS.enableTopP); setEnableMaxTokens(DEFAULT_SETTINGS.enableMaxTokens);
        setCustomParams(DEFAULT_SETTINGS.customParams); setAnswerDetailLevel(DEFAULT_SETTINGS.answerDetailLevel);
        setThoughtAutoCollapse(DEFAULT_SETTINGS.thoughtAutoCollapse); setSendShortcut(DEFAULT_SETTINGS.sendShortcut);
        setConfirmDeleteMessage(DEFAULT_SETTINGS.confirmDeleteMessage); setConfirmRegenerateMessage(DEFAULT_SETTINGS.confirmRegenerateMessage);
        setCodeCollapsible(DEFAULT_SETTINGS.codeCollapsible); setCodeWrappable(DEFAULT_SETTINGS.codeWrappable);
        setCodeShowLineNumbers(DEFAULT_SETTINGS.codeShowLineNumbers); setMathEngine(DEFAULT_SETTINGS.mathEngine);
        setMathEnableSingleDollar(DEFAULT_SETTINGS.mathEnableSingleDollar); setMessageStyle(DEFAULT_SETTINGS.messageStyle);
        setMessageFontSize(DEFAULT_SETTINGS.messageFontSize);
    };

    const addCustomParam = () => setCustomParams([...customParams, { name: '', type: 'string', value: '' }]);
    const updateCustomParam = (index, field, val) => {
        const updated = [...customParams];
        if (field === 'type') updated[index] = { ...updated[index], type: val, value: val === 'boolean' ? false : val === 'number' ? 0 : '' };
        else updated[index] = { ...updated[index], [field]: val };
        setCustomParams(updated);
    };
    const removeCustomParam = (index) => setCustomParams(customParams.filter((_, i) => i !== index));

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="fixed inset-0 bg-slate-950/25 z-50 flex items-center justify-center p-4 transition-all"
                onClick={onClose}
            >
                <motion.div
                    initial={{ scale: 0.95, opacity: 0, y: 15 }} animate={{ scale: 1, opacity: 1, y: 0 }} exit={{ scale: 0.95, opacity: 0, y: 10 }}
                    transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                    className="settings-solid settings-shell w-full max-w-[640px] max-h-[92vh] bg-[#f6f7f9] border border-white/80 overflow-hidden flex flex-col"
                    onClick={(e) => e.stopPropagation()}
                >
                    {/* 头部 */}
                    <div className="settings-chrome flex items-center px-6 py-5 sticky top-0 border-b border-gray-200 z-10">
                        <div className="flex items-center gap-3">
                            <button onClick={onClose} className="p-2 -ml-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-colors" title="返回设置中心" aria-label="返回设置中心">
                                <ChevronLeft className="w-5 h-5" />
                            </button>
                            <div className="w-10 h-10 bg-[#FFF4EF] rounded-[14px] border border-[#FFDCCF] flex items-center justify-center text-[#B85F47]">
                                <SlidersHorizontal className="w-5 h-5" />
                            </div>
                            <div>
                                <div className="text-[17px] font-bold text-gray-900 tracking-tight">对话设置</div>
                                <div className="text-[12px] text-gray-500 font-medium">调整模型生成参数，优化对话体验</div>
                            </div>
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto px-6 pb-6 pt-2 space-y-5 custom-scrollbar">
                        {/* ===== 生成参数 ===== (根据图1去除标题) */}
                        <div className="settings-card bg-white p-5 border border-gray-200/90 space-y-6">
                            {/* 上下文数 (无开关) */}
                            <SettingSliderWithInput
                                label="上下文数" tooltip="发送给模型的历史消息轮数。值越大模型记忆越多，但消耗更多 Token"
                                value={contextCount} onChange={(v) => setContextCount(Math.round(v))}
                                min={0} max={50} step={1} precision={0}
                            />

                            {/* 最大 Token 数 */}
                            <SettingToggleSlider
                                label="最大 Token 数" tooltip="限制模型单次回复的最大长度。关闭则由模型自行决定"
                                enabled={enableMaxTokens} onToggle={setEnableMaxTokens}
                                value={maxTokens} onChange={setMaxTokens}
                                min={512} max={32768} step={512} precision={0}
                            />

                            {/* 模型温度 */}
                            <SettingToggleSlider
                                label="模型温度" tooltip="控制回答的随机性。值越低越精确，值越高越有创造性"
                                enabled={enableTemperature} onToggle={setEnableTemperature}
                                value={temperature} onChange={setTemperature}
                                min={0} max={2} step={0.1} precision={1}
                            />

                            {/* Top-P */}
                            <SettingToggleSlider
                                label="Top-P" tooltip="核采样参数。控制候选词的概率范围，值越小回答越集中"
                                enabled={enableTopP} onToggle={setEnableTopP}
                                value={topP} onChange={setTopP}
                                min={0} max={1} step={0.05} precision={2}
                            />

                            {/* 回答详细度 */}
                            <div className="flex flex-col gap-3">
                                <div className="flex items-center gap-2">
                                    <span className="text-[14px] font-medium text-gray-800">回答详细度</span>
                                    <Tooltip text="控制回答展开程度。建议与最大 Token 数配合使用" />
                                </div>
                                <SettingsSegmentedControl
                                    ariaLabel="回答详细度"
                                    value={answerDetailLevel}
                                    onChange={setAnswerDetailLevel}
                                    options={[
                                        { value: 'concise', label: '简洁' },
                                        { value: 'standard', label: '标准' },
                                        { value: 'detailed', label: '详细' },
                                    ]}
                                />
                            </div>

                            {/* 流式输出 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-[14px] font-medium text-gray-800">流式输出</span>
                                    <Tooltip text="开启后回答会逐字显示，关闭则等待完整回答后一次性显示" />
                                </div>
                                <ToggleSwitch checked={streamOutput} onChange={setStreamOutput} />
                            </div>
                        </div>

                        {/* ===== 自定义参数 ===== */}
                        <div className="settings-card bg-white p-5 border border-gray-200/90">
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-2">
                                    <Code className="w-4 h-4 text-gray-400" />
                                    <h3 className="text-[14px] font-medium text-gray-800">自定义参数</h3>
                                    <Tooltip text="添加任意 key-value 参数直接传给 API" />
                                </div>
                                <button
                                    onClick={addCustomParam}
                                    className="accent-surface flex items-center gap-1 px-3 py-1.5 text-xs font-semibold rounded-xl transition-colors"
                                >
                                    <Plus className="w-3.5 h-3.5" /> 添加参数
                                </button>
                            </div>

                            <div className="space-y-3">
                                {customParams.length > 0 ? (
                                    <div className="space-y-2">
                                        {customParams.map((param, index) => (
                                            <CustomParamRow key={index} param={param} onChange={(field, val) => updateCustomParam(index, field, val)} onRemove={() => removeCustomParam(index)} />
                                        ))}
                                    </div>
                                ) : (
                                    <div className="text-xs text-gray-400 font-medium tracking-wide text-center py-6 bg-gray-50/50 rounded-2xl border border-dashed border-gray-200/80">
                                        暂无自定义参数
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* ===== 行为设置 ===== */}
                        <div className="settings-card bg-white p-5 border border-gray-200/90">
                            <div className="flex items-center gap-2 mb-4">
                                <SlidersHorizontal className="w-4 h-4 text-gray-400" />
                                <h3 className="text-[14px] font-medium text-gray-800">行为设置</h3>
                            </div>
                            
                            <div className="space-y-5">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <span className="text-[14px] font-medium text-gray-800">思考自动折叠</span>
                                        <Tooltip text="开启后，深度思考完成时自动折叠思考过程内容" />
                                    </div>
                                    <ToggleSwitch checked={thoughtAutoCollapse} onChange={setThoughtAutoCollapse} />
                                </div>

                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <span className="text-[14px] font-medium text-gray-800">发送快捷键</span>
                                    </div>
                                    <SettingsSegmentedControl
                                        ariaLabel="发送快捷键"
                                        value={sendShortcut}
                                        onChange={setSendShortcut}
                                        options={[
                                            { value: 'Enter', label: 'Enter' },
                                            { value: 'Ctrl+Enter', label: 'Ctrl+Enter' },
                                        ]}
                                        className="w-[180px] rounded-xl"
                                        buttonClassName="px-2 py-1 text-[13px] font-medium text-center rounded-[9px]"
                                        indicatorClassName="rounded-[9px]"
                                    />
                                </div>

                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <span className="text-[14px] font-medium text-gray-800">删除消息确认</span>
                                        <Tooltip text="删除消息前弹出确认对话框，防止误操作" />
                                    </div>
                                    <ToggleSwitch checked={confirmDeleteMessage} onChange={setConfirmDeleteMessage} />
                                </div>

                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <span className="text-[14px] font-medium text-gray-800">重新生成确认</span>
                                    </div>
                                    <ToggleSwitch checked={confirmRegenerateMessage} onChange={setConfirmRegenerateMessage} />
                                </div>
                            </div>
                        </div>

                        {/* ===== 界面与代码块 ===== */}
                        <div className="grid grid-cols-2 gap-4">
                            {/* 代码块设置 */}
                            <div className="settings-card bg-white p-5 border border-gray-200/90 space-y-4">
                                <div className="flex items-center gap-2 mb-2">
                                    <Code className="w-4 h-4 text-gray-400" />
                                    <h3 className="text-[14px] font-medium text-gray-800">代码块</h3>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-[13px] font-medium text-gray-700">折叠长代码</span>
                                    <ToggleSwitch checked={codeCollapsible} onChange={setCodeCollapsible} />
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-[13px] font-medium text-gray-700">自动换行</span>
                                    <ToggleSwitch checked={codeWrappable} onChange={setCodeWrappable} />
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-[13px] font-medium text-gray-700">显示行号</span>
                                    <ToggleSwitch checked={codeShowLineNumbers} onChange={setCodeShowLineNumbers} />
                                </div>
                            </div>

                            {/* 其他 (数学与样式) */}
                            <div className="settings-card bg-white p-5 border border-gray-200/90 space-y-4">
                                <div className="flex items-center gap-2 mb-2">
                                    <Type className="w-4 h-4 text-gray-400" />
                                    <h3 className="text-[14px] font-medium text-gray-800">外观</h3>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-[13px] font-medium text-gray-700">气泡样式</span>
                                    <ToggleSwitch checked={messageStyle === 'bubble'} onChange={(v) => setMessageStyle(v ? 'bubble' : 'plain')} />
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-[13px] font-medium text-gray-700">单$行内数学</span>
                                    <ToggleSwitch checked={mathEnableSingleDollar} onChange={setMathEnableSingleDollar} />
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-[13px] font-medium text-gray-700">字体大小</span>
                                    <div className="w-16">
                                        <NumberInput value={messageFontSize} onChange={setMessageFontSize} min={12} max={22} step={1} precision={0} />
                                    </div>
                                </div>
                            </div>
                        </div>

                    </div>
                    {/* 底部重置按钮 - 悬浮在底部 */}
                    <div className="settings-chrome p-4 border-t border-gray-200 flex justify-center">
                        <button
                            onClick={() => { if (confirm('确定要重置所有对话参数为默认值吗？')) resetChatSettings(); }}
                            className="w-full flex items-center justify-center gap-2 py-2.5 text-[13px] font-medium text-gray-500 hover:text-red-500 hover:bg-red-50/50 rounded-2xl transition-all"
                        >
                            <RotateCcw className="w-3.5 h-3.5" />
                            <span>恢复默认设置</span>
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
        <HelpCircle className="w-3.5 h-3.5 text-gray-300 hover:text-gray-400 cursor-help transition-colors" />
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-gray-800 text-white text-[11px] font-medium rounded-[8px] opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-normal text-center w-max max-w-[200px] shadow-lg z-50">
            {text}
        </div>
    </div>
);

/** 开关组件 — iOS 风格紫圆角 */
const ToggleSwitch = ({ checked, onChange }) => (
    <button
        onClick={() => onChange(!checked)}
        className={`relative w-[42px] h-[24px] rounded-full transition-colors duration-200 outline-none ${checked ? 'accent-control' : 'bg-gray-200'}`}
    >
        <div className={`absolute top-[2px] left-[2px] w-[20px] h-[20px] bg-white rounded-full shadow-sm transition-transform duration-200 ${checked ? 'translate-x-[18px]' : ''}`} />
    </button>
);

/**
 * 数字输入框组件 — 极简药丸风格
 */
const NumberInput = ({ value, onChange, min, max, step, precision, disabled }) => {
    const [localValue, setLocalValue] = useState(String(value));

    useEffect(() => {
        const formatted = precision > 0 ? Number(value).toFixed(precision) : String(Math.round(value));
        setLocalValue(formatted);
    }, [value, precision]);

    const handleBlur = () => {
        let num = parseFloat(localValue);
        if (isNaN(num)) num = value;
        num = Math.min(Math.max(num, min), max);
        const formatted = precision > 0 ? Number(num).toFixed(precision) : String(Math.round(num));
        setLocalValue(formatted);
        onChange(Number(formatted));
    };

    return (
        <input
            type="text" value={localValue} onChange={(e) => setLocalValue(e.target.value)} onBlur={handleBlur} disabled={disabled}
            className={`w-full text-center text-[13px] font-medium border rounded-[12px] px-2 py-1 outline-none transition-colors
                ${disabled ? 'bg-gray-50 text-gray-400 border-transparent cursor-not-allowed' : 'bg-white text-gray-700 border-gray-200 focus:border-[#ed8c68]/50 focus:ring-2 focus:ring-[#ed8c68]/10 hover:border-gray-300'}`}
        />
    );
};

/**
 * 现代滑块 + 左侧标题/右侧控件
 */
const SettingToggleSlider = ({ label, tooltip, enabled, onToggle, value, onChange, min, max, step, precision }) => {
    const pct = ((value - min) / (max - min)) * 100;
    const gc = '#ed8c68';

    return (
        <div className="flex flex-col gap-2.5">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="text-[14px] font-medium text-gray-800">{label}</span>
                    {tooltip && <Tooltip text={tooltip} />}
                </div>
                <div className="flex items-center gap-2.5">
                    {enabled !== undefined && <ToggleSwitch checked={enabled} onChange={onToggle} />}
                    <div className="w-[60px]">
                        <NumberInput value={value} onChange={onChange} min={min} max={max} step={step} precision={precision} disabled={enabled !== undefined && !enabled} />
                    </div>
                </div>
            </div>
            <div className="px-1 py-1">
                <input
                    type="range" min={min} max={max} step={step} value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    disabled={enabled !== undefined && !enabled}
                    className={`w-full h-[6px] rounded-full appearance-none bg-gray-100 ${enabled === false ? 'cursor-not-allowed opacity-40' : 'cursor-pointer'}`}
                    style={{
                        background: enabled !== false
                            ? `linear-gradient(to right, ${gc} 0%, ${gc} ${pct}%, #F3F4F6 ${pct}%, #F3F4F6 100%)`
                            : '#F3F4F6'
                    }}
                />
            </div>
        </div>
    );
};

/**
 * 无开关滑块控件
 */
const SettingSliderWithInput = ({ label, tooltip, value, onChange, min, max, step, precision }) => {
    const pct = ((value - min) / (max - min)) * 100;
    const gc = '#ed8c68';

    return (
        <div className="flex flex-col gap-2.5">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="text-[14px] font-medium text-gray-800">{label}</span>
                    {tooltip && <Tooltip text={tooltip} />}
                </div>
                <div className="w-[60px]">
                    <NumberInput value={value} onChange={onChange} min={min} max={max} step={step} precision={precision} disabled={false} />
                </div>
            </div>
            <div className="px-1 py-1">
                <input
                    type="range" min={min} max={max} step={step} value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    className="w-full h-[6px] rounded-full appearance-none bg-gray-100 cursor-pointer"
                    style={{ background: `linear-gradient(to right, ${gc} 0%, ${gc} ${pct}%, #F3F4F6 ${pct}%, #F3F4F6 100%)` }}
                />
            </div>
        </div>
    );
};

const CustomParamRow = ({ param, onChange, onRemove }) => {
    return (
        <div className="flex items-center gap-2 p-1.5 bg-gray-50/80 rounded-[14px] border border-gray-100/50">
            <input
                type="text" value={param.name} onChange={(e) => onChange('name', e.target.value)} placeholder="Key"
                className="flex-[2] text-[13px] border-none rounded-[10px] px-2 py-1.5 outline-none focus:bg-white focus:ring-2 focus:ring-[#ed8c68]/20 bg-transparent transition-all"
            />
            <div className="w-[1px] h-4 bg-gray-200"></div>
            <select
                value={param.type} onChange={(e) => onChange('type', e.target.value)}
                className="flex-[1] text-[12px] text-gray-500 font-medium border-none rounded-[10px] px-2 py-1.5 outline-none focus:bg-white focus:ring-2 focus:ring-[#ed8c68]/20 bg-transparent transition-all cursor-pointer appearance-none"
            >
                <option value="string">str</option>
                <option value="number">num</option>
                <option value="boolean">bool</option>
            </select>
            <div className="w-[1px] h-4 bg-gray-200"></div>
            {param.type === 'boolean' ? (
                <div className="flex-[2] flex justify-center py-1">
                    <ToggleSwitch checked={!!param.value} onChange={(v) => onChange('value', v)} />
                </div>
            ) : (
                <input
                    type={param.type === 'number' ? 'number' : 'text'}
                    value={param.value} onChange={(e) => onChange('value', param.type === 'number' ? Number(e.target.value) : e.target.value)} placeholder="Value"
                    className="flex-[2] text-[13px] border-none rounded-[10px] px-2 py-1.5 outline-none focus:bg-white focus:ring-2 focus:ring-[#ed8c68]/20 bg-transparent transition-all"
                />
            )}
            <button onClick={onRemove} className="p-1.5 mx-1 text-gray-300 hover:text-red-500 hover:bg-white rounded-lg transition-all">
                <Trash2 className="w-3.5 h-3.5" />
            </button>
        </div>
    );
};

export default ChatSettings;
