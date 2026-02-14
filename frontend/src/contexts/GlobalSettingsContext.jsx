import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';

const GlobalSettingsContext = createContext();

// 预设字体列表
export const PRESET_FONTS = [
    { id: 'inter', name: 'Inter', value: 'Inter, sans-serif', googleFont: 'Inter:wght@300;400;500;600;700' },
    { id: 'roboto', name: 'Roboto', value: 'Roboto, sans-serif', googleFont: 'Roboto:wght@300;400;500;700' },
    { id: 'noto-sans-sc', name: 'Noto Sans SC', value: '"Noto Sans SC", sans-serif', googleFont: 'Noto+Sans+SC:wght@300;400;500;700' },
    { id: 'source-han-sans', name: 'Source Han Sans', value: '"Source Han Sans SC", "Noto Sans SC", sans-serif', googleFont: 'Noto+Sans+SC:wght@300;400;500;700' },
    { id: 'poppins', name: 'Poppins', value: 'Poppins, sans-serif', googleFont: 'Poppins:wght@300;400;500;600;700' },
    { id: 'open-sans', name: 'Open Sans', value: '"Open Sans", sans-serif', googleFont: 'Open+Sans:wght@300;400;500;600;700' },
    { id: 'lato', name: 'Lato', value: 'Lato, sans-serif', googleFont: 'Lato:wght@300;400;700' },
    { id: 'montserrat', name: 'Montserrat', value: 'Montserrat, sans-serif', googleFont: 'Montserrat:wght@300;400;500;600;700' },
];

// 默认设置
const DEFAULT_SETTINGS = {
    fontFamily: 'inter',
    customFont: '',
    globalScale: 1.0,
    // 对话参数
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

export const GlobalSettingsProvider = ({ children }) => {
    const [fontFamily, setFontFamily] = useState(DEFAULT_SETTINGS.fontFamily);
    const [customFont, setCustomFont] = useState(DEFAULT_SETTINGS.customFont);
    const [globalScale, setGlobalScale] = useState(DEFAULT_SETTINGS.globalScale);
    const [maxTokens, setMaxTokens] = useState(DEFAULT_SETTINGS.maxTokens);
    const [temperature, setTemperature] = useState(DEFAULT_SETTINGS.temperature);
    const [topP, setTopP] = useState(DEFAULT_SETTINGS.topP);
    const [contextCount, setContextCount] = useState(DEFAULT_SETTINGS.contextCount);
    const [streamOutput, setStreamOutput] = useState(DEFAULT_SETTINGS.streamOutput);
    // 参数启用开关
    const [enableTemperature, setEnableTemperature] = useState(DEFAULT_SETTINGS.enableTemperature);
    const [enableTopP, setEnableTopP] = useState(DEFAULT_SETTINGS.enableTopP);
    const [enableMaxTokens, setEnableMaxTokens] = useState(DEFAULT_SETTINGS.enableMaxTokens);
    // 自定义参数
    const [customParams, setCustomParams] = useState(DEFAULT_SETTINGS.customParams);
    // 深度思考力度
    const [reasoningEffort, setReasoningEffort] = useState(DEFAULT_SETTINGS.reasoningEffort);
    // 记忆功能
    const [enableMemory, setEnableMemory] = useState(DEFAULT_SETTINGS.enableMemory);

    // 防抖保存相关 ref
    const debounceTimerRef = useRef(null);
    const pendingSettingsRef = useRef(null);

    // 从 localStorage 加载设置
    useEffect(() => {
        const loadSettings = () => {
            try {
                const saved = localStorage.getItem('globalSettings');
                if (saved) {
                    const settings = JSON.parse(saved);
                    setFontFamily(settings.fontFamily || DEFAULT_SETTINGS.fontFamily);
                    setCustomFont(settings.customFont || DEFAULT_SETTINGS.customFont);
                    setGlobalScale(settings.globalScale || DEFAULT_SETTINGS.globalScale);
                    if (settings.maxTokens !== undefined) setMaxTokens(settings.maxTokens);
                    if (settings.temperature !== undefined) setTemperature(settings.temperature);
                    if (settings.topP !== undefined) setTopP(settings.topP);
                    if (settings.contextCount !== undefined) setContextCount(settings.contextCount);
                    if (settings.streamOutput !== undefined) setStreamOutput(settings.streamOutput);
                    // 加载参数启用开关
                    if (settings.enableTemperature !== undefined) setEnableTemperature(settings.enableTemperature);
                    if (settings.enableTopP !== undefined) setEnableTopP(settings.enableTopP);
                    if (settings.enableMaxTokens !== undefined) setEnableMaxTokens(settings.enableMaxTokens);
                    // 加载自定义参数
                    if (settings.customParams !== undefined) setCustomParams(settings.customParams);
                    // 加载深度思考力度
                    if (settings.reasoningEffort !== undefined) setReasoningEffort(settings.reasoningEffort);
                    // 加载记忆功能开关
                    if (settings.enableMemory !== undefined) setEnableMemory(settings.enableMemory);
                }
            } catch (error) {
                console.error('Failed to load global settings:', error);
            }
        };
        loadSettings();
    }, []);

    // 防抖保存：立即将设置写入 pendingSettingsRef，500ms 后写入 localStorage
    const flushSave = useCallback(() => {
        if (pendingSettingsRef.current !== null) {
            try {
                localStorage.setItem('globalSettings', JSON.stringify(pendingSettingsRef.current));
            } catch (error) {
                console.error('保存全局设置失败:', error);
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

    // 监听所有设置变更，触发防抖保存
    useEffect(() => {
        const settings = {
            fontFamily,
            customFont,
            globalScale,
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
    }, [fontFamily, customFont, globalScale, maxTokens, temperature, topP, contextCount, streamOutput,
        enableTemperature, enableTopP, enableMaxTokens, customParams, reasoningEffort, enableMemory, debouncedSave]);

    // 组件卸载时 flush 未保存的数据
    useEffect(() => {
        return () => {
            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }
            flushSave();
        };
    }, [flushSave]);

    // 应用字体到 CSS
    useEffect(() => {
        const applyFont = () => {
            let fontValue;

            if (fontFamily === 'custom' && customFont) {
                // 使用自定义字体
                fontValue = `"${customFont}", sans-serif`;
                loadGoogleFont(customFont);
            } else {
                // 使用预设字体
                const preset = PRESET_FONTS.find(f => f.id === fontFamily);
                if (preset) {
                    fontValue = preset.value;
                    loadGoogleFont(preset.googleFont);
                } else {
                    fontValue = PRESET_FONTS[0].value; // 默认 Inter
                }
            }

            document.documentElement.style.setProperty('--global-font-family', fontValue);
        };

        applyFont();
    }, [fontFamily, customFont]);

    // 应用字体大小到 html 根元素
    useEffect(() => {
        // globalScale 作为字体缩放因子，1.0 = 16px 基准
        const baseFontSize = 16;
        const fontSize = Math.round(baseFontSize * globalScale);
        document.documentElement.style.fontSize = `${fontSize}px`;
        document.documentElement.style.setProperty('--global-scale', globalScale.toString());

        // 清除之前可能残留的 #root transform 和 body zoom
        const root = document.getElementById('root');
        if (root) {
            root.style.transform = '';
            root.style.transformOrigin = '';
            root.style.width = '';
            root.style.height = '';
        }
        document.body.style.zoom = '';
    }, [globalScale]);

    // 加载 Google Font
    const loadGoogleFont = (fontSpec) => {
        // 检查是否已经加载
        const existingLink = document.getElementById('google-fonts-global');

        // 构建 Google Fonts URL
        let fontUrl;
        if (fontSpec.includes(':')) {
            // 已经是完整的 font spec (e.g., 'Inter:wght@300;400;500')
            fontUrl = `https://fonts.googleapis.com/css2?family=${fontSpec}&display=swap`;
        } else {
            // 只是字体名称，使用默认权重
            const encodedName = fontSpec.replace(/\s+/g, '+');
            fontUrl = `https://fonts.googleapis.com/css2?family=${encodedName}:wght@300;400;500;600;700&display=swap`;
        }

        if (existingLink) {
            existingLink.href = fontUrl;
        } else {
            const link = document.createElement('link');
            link.id = 'google-fonts-global';
            link.rel = 'stylesheet';
            link.href = fontUrl;
            document.head.appendChild(link);
        }
    };

    // 重置设置
    const resetSettings = () => {
        setFontFamily(DEFAULT_SETTINGS.fontFamily);
        setCustomFont(DEFAULT_SETTINGS.customFont);
        setGlobalScale(DEFAULT_SETTINGS.globalScale);
        setMaxTokens(DEFAULT_SETTINGS.maxTokens);
        setTemperature(DEFAULT_SETTINGS.temperature);
        setTopP(DEFAULT_SETTINGS.topP);
        setContextCount(DEFAULT_SETTINGS.contextCount);
        setStreamOutput(DEFAULT_SETTINGS.streamOutput);
        setEnableTemperature(DEFAULT_SETTINGS.enableTemperature);
        setEnableTopP(DEFAULT_SETTINGS.enableTopP);
        setEnableMaxTokens(DEFAULT_SETTINGS.enableMaxTokens);
        setCustomParams(DEFAULT_SETTINGS.customParams);
        setReasoningEffort(DEFAULT_SETTINGS.reasoningEffort);
        setEnableMemory(DEFAULT_SETTINGS.enableMemory);
    };

    // 导出设置
    const exportSettings = () => {
        const settings = {
            fontFamily,
            customFont,
            globalScale,
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
            exportedAt: new Date().toISOString(),
        };
        return JSON.stringify(settings, null, 2);
    };

    // 导入设置
    const importSettings = (jsonString) => {
        try {
            const settings = JSON.parse(jsonString);
            if (settings.fontFamily !== undefined) setFontFamily(settings.fontFamily);
            if (settings.customFont !== undefined) setCustomFont(settings.customFont);
            if (settings.globalScale !== undefined) setGlobalScale(settings.globalScale);
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
            return true;
        } catch (error) {
            console.error('导入设置失败:', error);
            return false;
        }
    };

    // 获取当前字体显示名称
    const getCurrentFontName = () => {
        if (fontFamily === 'custom') {
            return customFont || '自定义字体';
        }
        const preset = PRESET_FONTS.find(f => f.id === fontFamily);
        return preset ? preset.name : 'Inter';
    };

    const value = {
        // 状态
        fontFamily,
        customFont,
        globalScale,
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
        setFontFamily,
        setCustomFont,
        setGlobalScale,
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
        resetSettings,
        exportSettings,
        importSettings,
        getCurrentFontName,
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

// Hook
export const useGlobalSettings = () => {
    const context = useContext(GlobalSettingsContext);
    if (!context) {
        throw new Error('useGlobalSettings must be used within GlobalSettingsProvider');
    }
    return context;
};

export default GlobalSettingsContext;
