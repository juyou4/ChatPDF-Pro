import React, { createContext, useContext, useCallback } from 'react';
import { FontSettingsProvider, useFontSettings, PRESET_FONTS, FONT_DEFAULT_SETTINGS } from './FontSettingsContext';
import { ChatParamsProvider, useChatParams, CHAT_PARAMS_DEFAULT_SETTINGS } from './ChatParamsContext';

// GlobalSettingsContext —— 聚合层
// 组合 FontSettingsContext 和 ChatParamsContext，保持向后兼容（需求 2.1）
// 新代码应优先使用 useFontSettings 或 useChatParams 实现细粒度订阅

const GlobalSettingsContext = createContext();

// 重新导出预设字体列表，保持向后兼容
export { PRESET_FONTS };

// 合并后的默认设置，保持向后兼容
const DEFAULT_SETTINGS = {
    ...FONT_DEFAULT_SETTINGS,
    ...CHAT_PARAMS_DEFAULT_SETTINGS,
};

export { DEFAULT_SETTINGS };

/**
 * 聚合层 Provider —— 组合 FontSettingsProvider 和 ChatParamsProvider
 * 内部嵌套子 Context Provider，外部使用方式不变
 */
export const GlobalSettingsProvider = ({ children }) => (
    <FontSettingsProvider>
        <ChatParamsProvider>
            <GlobalSettingsBridge>
                {children}
            </GlobalSettingsBridge>
        </ChatParamsProvider>
    </FontSettingsProvider>
);

/**
 * 内部桥接组件 —— 将两个子 Context 的值合并到 GlobalSettingsContext 中
 * 确保 useGlobalSettings 返回的接口与重构前完全一致
 */
const GlobalSettingsBridge = ({ children }) => {
    const fontSettings = useFontSettings();
    const chatParams = useChatParams();

    // 聚合重置：同时重置字体设置和对话参数
    const resetSettings = useCallback(() => {
        fontSettings.resetFontSettings();
        chatParams.resetChatParams();
    }, [fontSettings.resetFontSettings, chatParams.resetChatParams]);

    // 聚合导出：合并两个子 Context 的所有设置
    const exportSettings = useCallback(() => {
        const settings = {
            // 字体设置
            fontFamily: fontSettings.fontFamily,
            customFont: fontSettings.customFont,
            globalScale: fontSettings.globalScale,
            // 对话参数
            maxTokens: chatParams.maxTokens,
            temperature: chatParams.temperature,
            topP: chatParams.topP,
            contextCount: chatParams.contextCount,
            streamOutput: chatParams.streamOutput,
            enableTemperature: chatParams.enableTemperature,
            enableTopP: chatParams.enableTopP,
            enableMaxTokens: chatParams.enableMaxTokens,
            customParams: chatParams.customParams,
            reasoningEffort: chatParams.reasoningEffort,
            enableMemory: chatParams.enableMemory,
            exportedAt: new Date().toISOString(),
        };
        return JSON.stringify(settings, null, 2);
    }, [
        fontSettings.fontFamily, fontSettings.customFont, fontSettings.globalScale,
        chatParams.maxTokens, chatParams.temperature, chatParams.topP,
        chatParams.contextCount, chatParams.streamOutput,
        chatParams.enableTemperature, chatParams.enableTopP, chatParams.enableMaxTokens,
        chatParams.customParams, chatParams.reasoningEffort, chatParams.enableMemory,
    ]);

    // 聚合导入：将设置分发到对应的子 Context
    const importSettings = useCallback((jsonString) => {
        try {
            const settings = JSON.parse(jsonString);
            // 字体相关
            if (settings.fontFamily !== undefined) fontSettings.setFontFamily(settings.fontFamily);
            if (settings.customFont !== undefined) fontSettings.setCustomFont(settings.customFont);
            if (settings.globalScale !== undefined) fontSettings.setGlobalScale(settings.globalScale);
            // 对话参数相关
            if (settings.maxTokens !== undefined) chatParams.setMaxTokens(settings.maxTokens);
            if (settings.temperature !== undefined) chatParams.setTemperature(settings.temperature);
            if (settings.topP !== undefined) chatParams.setTopP(settings.topP);
            if (settings.contextCount !== undefined) chatParams.setContextCount(settings.contextCount);
            if (settings.streamOutput !== undefined) chatParams.setStreamOutput(settings.streamOutput);
            if (settings.enableTemperature !== undefined) chatParams.setEnableTemperature(settings.enableTemperature);
            if (settings.enableTopP !== undefined) chatParams.setEnableTopP(settings.enableTopP);
            if (settings.enableMaxTokens !== undefined) chatParams.setEnableMaxTokens(settings.enableMaxTokens);
            if (settings.customParams !== undefined) chatParams.setCustomParams(settings.customParams);
            if (settings.reasoningEffort !== undefined) chatParams.setReasoningEffort(settings.reasoningEffort);
            if (settings.enableMemory !== undefined) chatParams.setEnableMemory(settings.enableMemory);
            return true;
        } catch (error) {
            console.error('导入设置失败:', error);
            return false;
        }
    }, [
        fontSettings.setFontFamily, fontSettings.setCustomFont, fontSettings.setGlobalScale,
        chatParams.setMaxTokens, chatParams.setTemperature, chatParams.setTopP,
        chatParams.setContextCount, chatParams.setStreamOutput,
        chatParams.setEnableTemperature, chatParams.setEnableTopP, chatParams.setEnableMaxTokens,
        chatParams.setCustomParams, chatParams.setReasoningEffort, chatParams.setEnableMemory,
    ]);

    // 聚合 flushSave：同时 flush 两个子 Context
    const flushSave = useCallback(() => {
        fontSettings.flushSave();
        chatParams.flushSave();
    }, [fontSettings.flushSave, chatParams.flushSave]);

    // 合并所有值，保持与重构前完全一致的接口
    const value = {
        // 字体设置状态
        fontFamily: fontSettings.fontFamily,
        customFont: fontSettings.customFont,
        globalScale: fontSettings.globalScale,

        // 对话参数状态
        maxTokens: chatParams.maxTokens,
        temperature: chatParams.temperature,
        topP: chatParams.topP,
        contextCount: chatParams.contextCount,
        streamOutput: chatParams.streamOutput,
        enableTemperature: chatParams.enableTemperature,
        enableTopP: chatParams.enableTopP,
        enableMaxTokens: chatParams.enableMaxTokens,
        customParams: chatParams.customParams,
        reasoningEffort: chatParams.reasoningEffort,
        enableMemory: chatParams.enableMemory,

        // 字体设置方法
        setFontFamily: fontSettings.setFontFamily,
        setCustomFont: fontSettings.setCustomFont,
        setGlobalScale: fontSettings.setGlobalScale,

        // 对话参数设置方法
        setMaxTokens: chatParams.setMaxTokens,
        setTemperature: chatParams.setTemperature,
        setTopP: chatParams.setTopP,
        setContextCount: chatParams.setContextCount,
        setStreamOutput: chatParams.setStreamOutput,
        setEnableTemperature: chatParams.setEnableTemperature,
        setEnableTopP: chatParams.setEnableTopP,
        setEnableMaxTokens: chatParams.setEnableMaxTokens,
        setCustomParams: chatParams.setCustomParams,
        setReasoningEffort: chatParams.setReasoningEffort,
        setEnableMemory: chatParams.setEnableMemory,

        // 聚合工具方法
        resetSettings,
        exportSettings,
        importSettings,
        getCurrentFontName: fontSettings.getCurrentFontName,
        flushSave,

        // 常量
        PRESET_FONTS,
        DEFAULT_SETTINGS,
    };

    return (
        <GlobalSettingsContext.Provider value={value}>
            {children}
        </GlobalSettingsContext.Provider>
    );
};

/**
 * 聚合 Hook —— 返回所有设置（向后兼容）
 * 注意：新代码应优先使用 useFontSettings 或 useChatParams 实现细粒度订阅
 */
export const useGlobalSettings = () => {
    const context = useContext(GlobalSettingsContext);
    if (!context) {
        throw new Error('useGlobalSettings must be used within GlobalSettingsProvider');
    }
    return context;
};

export default GlobalSettingsContext;
