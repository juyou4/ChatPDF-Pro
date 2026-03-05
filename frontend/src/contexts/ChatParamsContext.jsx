import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';

// 对话参数 Context —— 仅管理对话相关参数设置
// 从 GlobalSettingsContext 中拆分出来，实现细粒度订阅，
// 避免对话参数变更触发字体设置消费者的重渲染（需求 2.1, 2.3）

const ChatParamsContext = createContext();

const normalizeMathEngine = (value) => {
    if (typeof value !== 'string') return CHAT_PARAMS_DEFAULT_SETTINGS.mathEngine;
    const normalized = value.trim().toLowerCase();
    if (normalized === 'katex') return 'KaTeX';
    if (normalized === 'mathjax') return 'MathJax';
    if (normalized === 'none' || normalized === 'off' || normalized === '关闭') return 'none';
    return CHAT_PARAMS_DEFAULT_SETTINGS.mathEngine;
};

// 对话参数默认设置
export const CHAT_PARAMS_DEFAULT_SETTINGS = {
    maxTokens: 8192,
    temperature: 0.7,
    topP: 1.0,
    contextCount: 5,
    streamOutput: true,
    // 参数启用开关
    enableTemperature: true,    // 默认启用
    enableTopP: false,          // 默认禁用（让模型用默认值）
    enableMaxTokens: true,      // 默认启用
    // 自定义参数
    customParams: [],           // [{name: string, value: string|number|boolean, type: 'string'|'number'|'boolean'}]
    // 深度思考力度
    reasoningEffort: 'off',     // 'off' | 'low' | 'medium' | 'high'
    // 回答详细度
    answerDetailLevel: 'standard', // 'concise' | 'standard' | 'detailed'
    // 记忆功能
    enableMemory: true,         // 是否启用智能记忆系统
    // 思考过程自动折叠
    thoughtAutoCollapse: true,  // 思考完成后自动折叠
    // 发送快捷键
    sendShortcut: 'Enter',      // 'Enter' | 'Ctrl+Enter'
    // 消息操作确认
    confirmDeleteMessage: true, // 删除消息前确认
    confirmRegenerateMessage: false, // 重新生成前确认
    // 代码块增强
    codeCollapsible: false,     // 代码块可折叠
    codeWrappable: true,        // 代码块自动换行
    codeShowLineNumbers: false, // 代码块显示行号
    // 数学公式引擎
    mathEngine: 'KaTeX',            // 'KaTeX' | 'MathJax' | 'none'
    mathEnableSingleDollar: true,   // 是否启用单 $ 行内公式
    // 消息样式
    messageStyle: 'plain',     // 'plain' | 'bubble'
    // 消息字体大小
    messageFontSize: 14,        // 12-22px
};

export const ChatParamsProvider = ({ children }) => {
    const [maxTokens, setMaxTokens] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.maxTokens);
    const [temperature, setTemperature] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.temperature);
    const [topP, setTopP] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.topP);
    const [contextCount, setContextCount] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.contextCount);
    const [streamOutput, setStreamOutput] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.streamOutput);
    // 参数启用开关
    const [enableTemperature, setEnableTemperature] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.enableTemperature);
    const [enableTopP, setEnableTopP] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.enableTopP);
    const [enableMaxTokens, setEnableMaxTokens] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.enableMaxTokens);
    // 自定义参数
    const [customParams, setCustomParams] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.customParams);
    // 深度思考力度
    const [reasoningEffort, setReasoningEffort] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.reasoningEffort);
    // 回答详细度
    const [answerDetailLevel, setAnswerDetailLevel] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.answerDetailLevel);
    // 记忆功能
    const [enableMemory, setEnableMemory] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.enableMemory);
    // 思考过程自动折叠
    const [thoughtAutoCollapse, setThoughtAutoCollapse] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.thoughtAutoCollapse);
    // 发送快捷键
    const [sendShortcut, setSendShortcut] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.sendShortcut);
    // 消息操作确认
    const [confirmDeleteMessage, setConfirmDeleteMessage] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.confirmDeleteMessage);
    const [confirmRegenerateMessage, setConfirmRegenerateMessage] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.confirmRegenerateMessage);
    // 代码块增强
    const [codeCollapsible, setCodeCollapsible] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.codeCollapsible);
    const [codeWrappable, setCodeWrappable] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.codeWrappable);
    const [codeShowLineNumbers, setCodeShowLineNumbers] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.codeShowLineNumbers);
    // 数学公式引擎
    const [mathEngine, setMathEngine] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.mathEngine);
    const [mathEnableSingleDollar, setMathEnableSingleDollar] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.mathEnableSingleDollar);
    // 消息样式
    const [messageStyle, setMessageStyle] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.messageStyle);
    // 消息字体大小
    const [messageFontSize, setMessageFontSize] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.messageFontSize);

    // 防抖保存相关 ref
    const debounceTimerRef = useRef(null);
    const pendingSettingsRef = useRef(null);

    // 从 localStorage 加载对话参数设置
    useEffect(() => {
        try {
            const saved = localStorage.getItem('chatParamsSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                if (settings.maxTokens !== undefined) setMaxTokens(settings.maxTokens);
                if (settings.temperature !== undefined) setTemperature(settings.temperature);
                if (settings.topP !== undefined) setTopP(settings.topP);
                if (settings.contextCount !== undefined) setContextCount(settings.contextCount);
                if (settings.streamOutput !== undefined) setStreamOutput(settings.streamOutput);
                if (settings.enableTemperature !== undefined) setEnableTemperature(settings.enableTemperature);
                if (settings.enableTopP !== undefined) setEnableTopP(settings.enableTopP);
                if (settings.enableMaxTokens !== undefined) setEnableMaxTokens(settings.enableMaxTokens);
                if (settings.customParams !== undefined) setCustomParams(settings.customParams);
                if (settings.reasoningEffort !== undefined) setReasoningEffort(settings.reasoningEffort);
                if (settings.answerDetailLevel !== undefined) setAnswerDetailLevel(settings.answerDetailLevel);
                if (settings.enableMemory !== undefined) setEnableMemory(settings.enableMemory);
                if (settings.thoughtAutoCollapse !== undefined) setThoughtAutoCollapse(settings.thoughtAutoCollapse);
                if (settings.sendShortcut !== undefined) setSendShortcut(settings.sendShortcut);
                if (settings.confirmDeleteMessage !== undefined) setConfirmDeleteMessage(settings.confirmDeleteMessage);
                if (settings.confirmRegenerateMessage !== undefined) setConfirmRegenerateMessage(settings.confirmRegenerateMessage);
                if (settings.codeCollapsible !== undefined) setCodeCollapsible(settings.codeCollapsible);
                if (settings.codeWrappable !== undefined) setCodeWrappable(settings.codeWrappable);
                if (settings.codeShowLineNumbers !== undefined) setCodeShowLineNumbers(settings.codeShowLineNumbers);
                if (settings.mathEngine !== undefined) setMathEngine(normalizeMathEngine(settings.mathEngine));
                if (settings.mathEnableSingleDollar !== undefined) setMathEnableSingleDollar(settings.mathEnableSingleDollar);
                if (settings.messageStyle !== undefined) setMessageStyle(settings.messageStyle);
                if (settings.messageFontSize !== undefined) setMessageFontSize(settings.messageFontSize);
            } else {
                // 兼容旧版：从 globalSettings 中迁移对话参数
                const globalSaved = localStorage.getItem('globalSettings');
                if (globalSaved) {
                    const globalSettings = JSON.parse(globalSaved);
                    if (globalSettings.maxTokens !== undefined) setMaxTokens(globalSettings.maxTokens);
                    if (globalSettings.temperature !== undefined) setTemperature(globalSettings.temperature);
                    if (globalSettings.topP !== undefined) setTopP(globalSettings.topP);
                    if (globalSettings.contextCount !== undefined) setContextCount(globalSettings.contextCount);
                    if (globalSettings.streamOutput !== undefined) setStreamOutput(globalSettings.streamOutput);
                    if (globalSettings.enableTemperature !== undefined) setEnableTemperature(globalSettings.enableTemperature);
                    if (globalSettings.enableTopP !== undefined) setEnableTopP(globalSettings.enableTopP);
                    if (globalSettings.enableMaxTokens !== undefined) setEnableMaxTokens(globalSettings.enableMaxTokens);
                    if (globalSettings.customParams !== undefined) setCustomParams(globalSettings.customParams);
                    if (globalSettings.reasoningEffort !== undefined) setReasoningEffort(globalSettings.reasoningEffort);
                    if (globalSettings.answerDetailLevel !== undefined) setAnswerDetailLevel(globalSettings.answerDetailLevel);
                    if (globalSettings.enableMemory !== undefined) setEnableMemory(globalSettings.enableMemory);
                    if (globalSettings.thoughtAutoCollapse !== undefined) setThoughtAutoCollapse(globalSettings.thoughtAutoCollapse);
                    if (globalSettings.sendShortcut !== undefined) setSendShortcut(globalSettings.sendShortcut);
                    if (globalSettings.confirmDeleteMessage !== undefined) setConfirmDeleteMessage(globalSettings.confirmDeleteMessage);
                    if (globalSettings.confirmRegenerateMessage !== undefined) setConfirmRegenerateMessage(globalSettings.confirmRegenerateMessage);
                    if (globalSettings.codeCollapsible !== undefined) setCodeCollapsible(globalSettings.codeCollapsible);
                    if (globalSettings.codeWrappable !== undefined) setCodeWrappable(globalSettings.codeWrappable);
                    if (globalSettings.codeShowLineNumbers !== undefined) setCodeShowLineNumbers(globalSettings.codeShowLineNumbers);
                    if (globalSettings.mathEngine !== undefined) setMathEngine(normalizeMathEngine(globalSettings.mathEngine));
                    if (globalSettings.mathEnableSingleDollar !== undefined) setMathEnableSingleDollar(globalSettings.mathEnableSingleDollar);
                    if (globalSettings.messageStyle !== undefined) setMessageStyle(globalSettings.messageStyle);
                    if (globalSettings.messageFontSize !== undefined) setMessageFontSize(globalSettings.messageFontSize);
                }
            }
        } catch (error) {
            console.error('加载对话参数设置失败:', error);
        }
    }, []);

    // 防抖保存到 localStorage
    const flushSave = useCallback(() => {
        if (pendingSettingsRef.current !== null) {
            try {
                localStorage.setItem('chatParamsSettings', JSON.stringify(pendingSettingsRef.current));
            } catch (error) {
                console.error('保存对话参数设置失败:', error);
            }
            pendingSettingsRef.current = null;
        }
    }, []);

    const debouncedSave = useCallback((settings) => {
        pendingSettingsRef.current = settings;
        if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
        }
        debounceTimerRef.current = setTimeout(() => {
            flushSave();
            debounceTimerRef.current = null;
        }, 500);
    }, [flushSave]);

    // 监听对话参数变更，触发防抖保存
    useEffect(() => {
        const settings = {
            maxTokens,
            temperature,
            topP,
            contextCount,
            streamOutput,
            enableTemperature,
            enableTopP,
            enableMaxTokens,
            customParams,
            reasoningEffort,
            answerDetailLevel,
            enableMemory,
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
        };
        debouncedSave(settings);
    }, [maxTokens, temperature, topP, contextCount, streamOutput,
        enableTemperature, enableTopP, enableMaxTokens, customParams,
        reasoningEffort, answerDetailLevel, enableMemory, thoughtAutoCollapse, sendShortcut,
        confirmDeleteMessage, confirmRegenerateMessage,
        codeCollapsible, codeWrappable, codeShowLineNumbers,
        mathEngine, mathEnableSingleDollar,
        messageStyle, messageFontSize, debouncedSave]);

    // 组件卸载时 flush 未保存的数据 + beforeunload 保护
    useEffect(() => {
        const handleBeforeUnload = () => flushSave();
        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }
            flushSave();
        };
    }, [flushSave]);

    // 重置对话参数
    const resetChatParams = useCallback(() => {
        setMaxTokens(CHAT_PARAMS_DEFAULT_SETTINGS.maxTokens);
        setTemperature(CHAT_PARAMS_DEFAULT_SETTINGS.temperature);
        setTopP(CHAT_PARAMS_DEFAULT_SETTINGS.topP);
        setContextCount(CHAT_PARAMS_DEFAULT_SETTINGS.contextCount);
        setStreamOutput(CHAT_PARAMS_DEFAULT_SETTINGS.streamOutput);
        setEnableTemperature(CHAT_PARAMS_DEFAULT_SETTINGS.enableTemperature);
        setEnableTopP(CHAT_PARAMS_DEFAULT_SETTINGS.enableTopP);
        setEnableMaxTokens(CHAT_PARAMS_DEFAULT_SETTINGS.enableMaxTokens);
        setCustomParams(CHAT_PARAMS_DEFAULT_SETTINGS.customParams);
        setReasoningEffort(CHAT_PARAMS_DEFAULT_SETTINGS.reasoningEffort);
        setAnswerDetailLevel(CHAT_PARAMS_DEFAULT_SETTINGS.answerDetailLevel);
        setEnableMemory(CHAT_PARAMS_DEFAULT_SETTINGS.enableMemory);
        setThoughtAutoCollapse(CHAT_PARAMS_DEFAULT_SETTINGS.thoughtAutoCollapse);
        setSendShortcut(CHAT_PARAMS_DEFAULT_SETTINGS.sendShortcut);
        setConfirmDeleteMessage(CHAT_PARAMS_DEFAULT_SETTINGS.confirmDeleteMessage);
        setConfirmRegenerateMessage(CHAT_PARAMS_DEFAULT_SETTINGS.confirmRegenerateMessage);
        setCodeCollapsible(CHAT_PARAMS_DEFAULT_SETTINGS.codeCollapsible);
        setCodeWrappable(CHAT_PARAMS_DEFAULT_SETTINGS.codeWrappable);
        setCodeShowLineNumbers(CHAT_PARAMS_DEFAULT_SETTINGS.codeShowLineNumbers);
        setMathEngine(CHAT_PARAMS_DEFAULT_SETTINGS.mathEngine);
        setMathEnableSingleDollar(CHAT_PARAMS_DEFAULT_SETTINGS.mathEnableSingleDollar);
        setMessageStyle(CHAT_PARAMS_DEFAULT_SETTINGS.messageStyle);
        setMessageFontSize(CHAT_PARAMS_DEFAULT_SETTINGS.messageFontSize);
    }, []);

    const value = {
        // 状态
        maxTokens,
        temperature,
        topP,
        contextCount,
        streamOutput,
        enableTemperature,
        enableTopP,
        enableMaxTokens,
        customParams,
        reasoningEffort,
        answerDetailLevel,
        enableMemory,
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

        // 设置方法
        setMaxTokens,
        setTemperature,
        setTopP,
        setContextCount,
        setStreamOutput,
        setEnableTemperature,
        setEnableTopP,
        setEnableMaxTokens,
        setCustomParams,
        setReasoningEffort,
        setAnswerDetailLevel,
        setEnableMemory,
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

        // 工具方法
        resetChatParams,
        flushSave,

        // 常量
        CHAT_PARAMS_DEFAULT_SETTINGS,
    };

    return (
        <ChatParamsContext.Provider value={value}>
            {children}
        </ChatParamsContext.Provider>
    );
};

/**
 * 对话参数 Hook —— 仅订阅对话相关参数设置
 * 使用此 Hook 的组件不会因字体设置变更而重渲染
 */
export const useChatParams = () => {
    const context = useContext(ChatParamsContext);
    if (!context) {
        throw new Error('useChatParams 必须在 ChatParamsProvider 内部使用');
    }
    return context;
};

export default ChatParamsContext;
