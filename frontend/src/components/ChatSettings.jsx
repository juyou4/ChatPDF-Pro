import React, { useState, useEffect } from 'react';
import { X, SlidersHorizontal, HelpCircle, RotateCcw, Plus, Trash2, Code, MessageSquare, Type, Sigma } from 'lucide-react';
import { useChatParams, CHAT_PARAMS_DEFAULT_SETTINGS } from '../contexts/ChatParamsContext';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * 对话设置面板
 * 参考 cherry-studio 风格，包含：模型温度、Top-P、上下文数、最大 Token 数、流式输出、自定义参数
 * 每个参数行：标签 + 问号提示 | 开关（右侧）
 * 开关启用时显示：滑块（80%宽度）+ 数字输入框（20%宽度）
 */
const ChatSettings = ({ isOpen, onClose }) => {
    // 使用细粒度 Hook 订阅对话参数（需求 2.3）
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
        thoughtAutoCollapse,
        sendShortcut,
        confirmDeleteMessage,
        confirmRegenerateMessage,
        codeCollapsible,
        codeWrappable,
        codeShowLineNumbers,
        mathEngine,
        mathEnableSingleDollar,
        messageStyle,
        messageFontSize,
        setMaxTokens,
        setTemperature,
        setTopP,
        setContextCount,
        setStreamOutput,
        setEnableTemperature,
        setEnableTopP,
        setEnableMaxTokens,
        setCustomParams,
        setThoughtAutoCollapse,
        setSendShortcut,
        setConfirmDeleteMessage,
        setConfirmRegenerateMessage,
        setCodeCollapsible,
        setCodeWrappable,
        setCodeShowLineNumbers,
        setMathEngine,
        setMathEnableSingleDollar,
        setMessageStyle,
        setMessageFontSize,
    } = useChatParams();

    const DEFAULT_SETTINGS = CHAT_PARAMS_DEFAULT_SETTINGS;

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
        setThoughtAutoCollapse(DEFAULT_SETTINGS.thoughtAutoCollapse);
        setSendShortcut(DEFAULT_SETTINGS.sendShortcut);
        setConfirmDeleteMessage(DEFAULT_SETTINGS.confirmDeleteMessage);
        setConfirmRegenerateMessage(DEFAULT_SETTINGS.confirmRegenerateMessage);
        setCodeCollapsible(DEFAULT_SETTINGS.codeCollapsible);
        setCodeWrappable(DEFAULT_SETTINGS.codeWrappable);
        setCodeShowLineNumbers(DEFAULT_SETTINGS.codeShowLineNumbers);
        setMathEngine(DEFAULT_SETTINGS.mathEngine);
        setMathEnableSingleDollar(DEFAULT_SETTINGS.mathEnableSingleDollar);
        setMessageStyle(DEFAULT_SETTINGS.messageStyle);
        setMessageFontSize(DEFAULT_SETTINGS.messageFontSize);
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
                    <div className="sticky top-0 bg-white/90 backdrop-blur-md border-b border-gray-200 px-6 py-4 flex items-center justify-between z-10">
                        <div className="flex items-center gap-3">
                            <div className="w-9 h-9 bg-gray-100 rounded-lg flex items-center justify-center">
                                <SlidersHorizontal className="w-5 h-5 text-gray-600" />
                            </div>
                            <div>
                                <h2 className="text-xl font-semibold text-gray-900">对话设置</h2>
                                <p className="text-xs text-gray-500">调整模型生成参数</p>
                            </div>
                        </div>
                        <button onClick={onClose} className="p-1.5 hover:bg-gray-100 rounded-md transition-colors">
                            <X className="w-5 h-5 text-gray-500" />
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

                        <div className="border-t border-gray-200"></div>

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
                                    className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
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

                        <div className="border-t border-gray-200"></div>

                        {/* ===== 行为设置 ===== */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 mb-1">
                                <SlidersHorizontal className="w-4 h-4 text-gray-500" />
                                <span className="text-sm font-bold text-gray-700">行为设置</span>
                            </div>

                            {/* 思考完成后自动折叠 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">思考自动折叠</span>
                                    <Tooltip text="开启后，深度思考完成时自动折叠思考过程内容" />
                                </div>
                                <ToggleSwitch checked={thoughtAutoCollapse} onChange={setThoughtAutoCollapse} />
                            </div>

                            <div className="border-t border-gray-100"></div>

                            {/* 发送快捷键 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">发送快捷键</span>
                                    <Tooltip text="选择发送消息的快捷键方式" />
                                </div>
                                <div className="flex gap-1">
                                    {['Enter', 'Ctrl+Enter'].map((key) => (
                                        <button
                                            key={key}
                                            onClick={() => setSendShortcut(key)}
                                            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                                                sendShortcut === key
                                                    ? 'bg-gray-700 text-white'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                            }`}
                                        >
                                            {key}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="border-t border-gray-100"></div>

                            {/* 删除消息确认 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">删除消息确认</span>
                                    <Tooltip text="删除消息前弹出确认对话框，防止误操作" />
                                </div>
                                <ToggleSwitch checked={confirmDeleteMessage} onChange={setConfirmDeleteMessage} />
                            </div>

                            <div className="border-t border-gray-100"></div>

                            {/* 重新生成确认 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">重新生成确认</span>
                                    <Tooltip text="重新生成回答前弹出确认对话框" />
                                </div>
                                <ToggleSwitch checked={confirmRegenerateMessage} onChange={setConfirmRegenerateMessage} />
                            </div>
                        </div>

                        <div className="border-t border-gray-200"></div>

                        {/* ===== 代码块设置 ===== */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 mb-1">
                                <Code className="w-4 h-4 text-gray-500" />
                                <span className="text-sm font-bold text-gray-700">代码块设置</span>
                            </div>

                            {/* 代码块可折叠 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">代码块折叠</span>
                                    <Tooltip text="允许折叠/展开代码块，方便浏览长回答" />
                                </div>
                                <ToggleSwitch checked={codeCollapsible} onChange={setCodeCollapsible} />
                            </div>

                            <div className="border-t border-gray-100"></div>

                            {/* 代码自动换行 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">代码自动换行</span>
                                    <Tooltip text="代码块内容超出宽度时自动换行，而非水平滚动" />
                                </div>
                                <ToggleSwitch checked={codeWrappable} onChange={setCodeWrappable} />
                            </div>

                            <div className="border-t border-gray-100"></div>

                            {/* 代码行号 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">显示行号</span>
                                    <Tooltip text="在代码块左侧显示行号" />
                                </div>
                                <ToggleSwitch checked={codeShowLineNumbers} onChange={setCodeShowLineNumbers} />
                            </div>
                        </div>

                        <div className="border-t border-gray-200"></div>

                        {/* ===== 数学公式设置 ===== */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 mb-1">
                                <Sigma className="w-4 h-4 text-gray-500" />
                                <span className="text-sm font-bold text-gray-700">数学公式</span>
                            </div>

                            {/* 数学引擎 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">渲染引擎</span>
                                    <Tooltip text="选择数学公式渲染引擎。KaTeX 速度快，MathJax 兼容性好，关闭则不渲染公式" />
                                </div>
                                <div className="flex gap-1">
                                    {[{ value: 'KaTeX', label: 'KaTeX' }, { value: 'MathJax', label: 'MathJax' }, { value: 'none', label: '关闭' }].map(({ value, label }) => (
                                        <button
                                            key={value}
                                            onClick={() => setMathEngine(value)}
                                            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                                                mathEngine === value
                                                    ? 'bg-gray-700 text-white'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                            }`}
                                        >
                                            {label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="border-t border-gray-100"></div>

                            {/* 单 $ 行内公式 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">单 $ 行内公式</span>
                                    <Tooltip text="启用后，$...$ 会被识别为行内公式。关闭可避免与美元符号冲突" />
                                </div>
                                <ToggleSwitch checked={mathEnableSingleDollar} onChange={setMathEnableSingleDollar} />
                            </div>
                        </div>

                        <div className="border-t border-gray-200"></div>

                        {/* ===== 消息显示设置 ===== */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 mb-1">
                                <MessageSquare className="w-4 h-4 text-gray-500" />
                                <span className="text-sm font-bold text-gray-700">消息显示</span>
                            </div>

                            {/* 消息样式 */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-semibold text-gray-800">消息样式</span>
                                    <Tooltip text="选择消息的显示风格" />
                                </div>
                                <div className="flex gap-1">
                                    {[{ value: 'plain', label: '平铺' }, { value: 'bubble', label: '气泡' }].map(({ value, label }) => (
                                        <button
                                            key={value}
                                            onClick={() => setMessageStyle(value)}
                                            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                                                messageStyle === value
                                                    ? 'bg-gray-700 text-white'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                            }`}
                                        >
                                            {label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="border-t border-gray-100"></div>

                            {/* 消息字体大小 */}
                            <SettingSliderWithInput
                                label="消息字体大小"
                                tooltip="调整对话消息的字体大小（12-22px）"
                                value={messageFontSize}
                                onChange={(v) => setMessageFontSize(Math.round(v))}
                                min={12} max={22} step={1}
                                precision={0}
                                color="emerald"
                            />
                        </div>

                        <div className="border-t border-gray-200"></div>

                        {/* 重置按钮 */}
                        <button
                            onClick={() => { if (confirm('确定要重置所有对话参数为默认值吗？')) resetChatSettings(); }}
                            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 border border-gray-200 rounded-md transition-colors"
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

/** 开关组件 — 简约中性风格 */
const ToggleSwitch = ({ checked, onChange }) => (
    <button
        onClick={() => onChange(!checked)}
        className={`relative w-11 h-6 rounded-full transition-colors ${checked ? 'bg-gray-700' : 'bg-gray-300'}`}
    >
        <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform ${checked ? 'translate-x-5' : ''}`} />
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
            className={`w-full text-center text-sm font-mono border rounded-md px-2 py-1.5 outline-none transition-colors
                ${disabled
                    ? 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed'
                    : 'bg-white text-gray-800 border-gray-300 focus:border-gray-500 focus:ring-1 focus:ring-gray-200'
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
    // 简约风格：统一使用中性灰色
    const gc = '#6B7280'; // gray-500

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
    // 简约风格：统一使用中性灰色
    const gc = '#6B7280'; // gray-500

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
                className="flex-[2] text-sm border border-gray-300 rounded-md px-2 py-1.5 outline-none focus:border-gray-500 focus:ring-1 focus:ring-gray-200 bg-white"
            />
            {/* 类型选择 */}
            <select
                value={param.type}
                onChange={(e) => onChange('type', e.target.value)}
                className="flex-[1] text-sm border border-gray-300 rounded-md px-2 py-1.5 outline-none focus:border-gray-500 focus:ring-1 focus:ring-gray-200 bg-white cursor-pointer"
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
                    className="flex-[2] text-sm border border-gray-300 rounded-md px-2 py-1.5 outline-none focus:border-gray-500 focus:ring-1 focus:ring-gray-200 bg-white"
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
