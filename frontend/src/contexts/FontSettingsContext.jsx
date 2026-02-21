import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';

// 字体设置 Context —— 仅管理字体和缩放相关设置
// 从 GlobalSettingsContext 中拆分出来，实现细粒度订阅，
// 避免字体变更触发对话参数消费者的重渲染（需求 2.1, 2.2）

const FontSettingsContext = createContext();

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

// 字体相关默认设置
export const FONT_DEFAULT_SETTINGS = {
    fontFamily: 'inter',
    customFont: '',
    globalScale: 1.0,
};

/**
 * 加载 Google Font
 * @param {string} fontSpec - 字体规格，如 'Inter:wght@300;400;500' 或纯字体名称
 */
const loadGoogleFont = (fontSpec) => {
    // 检查是否已经加载
    const existingLink = document.getElementById('google-fonts-global');

    // 构建 Google Fonts URL
    let fontUrl;
    if (fontSpec.includes(':')) {
        // 已经是完整的 font spec (例如 'Inter:wght@300;400;500')
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

export const FontSettingsProvider = ({ children }) => {
    const [fontFamily, setFontFamily] = useState(FONT_DEFAULT_SETTINGS.fontFamily);
    const [customFont, setCustomFont] = useState(FONT_DEFAULT_SETTINGS.customFont);
    const [globalScale, setGlobalScale] = useState(FONT_DEFAULT_SETTINGS.globalScale);

    // 防抖保存相关 ref
    const debounceTimerRef = useRef(null);
    const pendingSettingsRef = useRef(null);

    // 从 localStorage 加载字体设置
    useEffect(() => {
        try {
            const saved = localStorage.getItem('fontSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                if (settings.fontFamily !== undefined) setFontFamily(settings.fontFamily);
                if (settings.customFont !== undefined) setCustomFont(settings.customFont);
                if (settings.globalScale !== undefined) setGlobalScale(settings.globalScale);
            } else {
                // 兼容旧版：从 globalSettings 中迁移字体设置
                const globalSaved = localStorage.getItem('globalSettings');
                if (globalSaved) {
                    const globalSettings = JSON.parse(globalSaved);
                    if (globalSettings.fontFamily !== undefined) setFontFamily(globalSettings.fontFamily);
                    if (globalSettings.customFont !== undefined) setCustomFont(globalSettings.customFont);
                    if (globalSettings.globalScale !== undefined) setGlobalScale(globalSettings.globalScale);
                }
            }
        } catch (error) {
            console.error('加载字体设置失败:', error);
        }
    }, []);

    // 防抖保存到 localStorage
    const flushSave = useCallback(() => {
        if (pendingSettingsRef.current !== null) {
            try {
                localStorage.setItem('fontSettings', JSON.stringify(pendingSettingsRef.current));
            } catch (error) {
                console.error('保存字体设置失败:', error);
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

    // 监听字体设置变更，触发防抖保存
    useEffect(() => {
        const settings = { fontFamily, customFont, globalScale };
        debouncedSave(settings);
    }, [fontFamily, customFont, globalScale, debouncedSave]);

    // 组件卸载时 flush 未保存的数据
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

    // 应用字体到 CSS 变量
    useEffect(() => {
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
    }, [fontFamily, customFont]);

    // 应用缩放到 html 根元素
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

    // 获取当前字体显示名称
    const getCurrentFontName = useCallback(() => {
        if (fontFamily === 'custom') {
            return customFont || '自定义字体';
        }
        const preset = PRESET_FONTS.find(f => f.id === fontFamily);
        return preset ? preset.name : 'Inter';
    }, [fontFamily, customFont]);

    // 重置字体设置
    const resetFontSettings = useCallback(() => {
        setFontFamily(FONT_DEFAULT_SETTINGS.fontFamily);
        setCustomFont(FONT_DEFAULT_SETTINGS.customFont);
        setGlobalScale(FONT_DEFAULT_SETTINGS.globalScale);
    }, []);

    const value = {
        // 状态
        fontFamily,
        customFont,
        globalScale,

        // 设置方法
        setFontFamily,
        setCustomFont,
        setGlobalScale,

        // 工具方法
        getCurrentFontName,
        resetFontSettings,
        flushSave,

        // 常量
        PRESET_FONTS,
        FONT_DEFAULT_SETTINGS,
    };

    return (
        <FontSettingsContext.Provider value={value}>
            {children}
        </FontSettingsContext.Provider>
    );
};

/**
 * 字体设置 Hook —— 仅订阅字体和缩放相关设置
 * 使用此 Hook 的组件不会因对话参数变更而重渲染
 */
export const useFontSettings = () => {
    const context = useContext(FontSettingsContext);
    if (!context) {
        throw new Error('useFontSettings 必须在 FontSettingsProvider 内部使用');
    }
    return context;
};

export default FontSettingsContext;
