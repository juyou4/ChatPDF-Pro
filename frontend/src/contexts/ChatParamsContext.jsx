import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';

// 对话参数 Context —— 仅管理对话相关参数设置
// 从 GlobalSettingsContext 中拆分出来，实现细粒度订阅，
// 避免对话参数变更触发字体设置消费者的重渲染（需求 2.1, 2.3）

const ChatParamsContext = createContext();

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
    // 记忆功能
    enableMemory: true,         // 是否启用智能记忆系统
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
    // 记忆功能
    const [enableMemory, setEnableMemory] = useState(CHAT_PARAMS_DEFAULT_SETTINGS.enableMemory);

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
                if (settings.enableMemory !== undefined) setEnableMemory(settings.enableMemory);
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
                    if (globalSettings.enableMemory !== undefined) setEnableMemory(globalSettings.enableMemory);
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
            enableMemory,
        };
        debouncedSave(settings);
    }, [maxTokens, temperature, topP, contextCount, streamOutput,
        enableTemperature, enableTopP, enableMaxTokens, customParams,
        reasoningEffort, enableMemory, debouncedSave]);

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
        setEnableMemory(CHAT_PARAMS_DEFAULT_SETTINGS.enableMemory);
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
        enableMemory,

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
        setEnableMemory,

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
