import React, { createContext, useContext, useState, useEffect } from 'react';

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
};

export const GlobalSettingsProvider = ({ children }) => {
    const [fontFamily, setFontFamily] = useState(DEFAULT_SETTINGS.fontFamily);
    const [customFont, setCustomFont] = useState(DEFAULT_SETTINGS.customFont);
    const [globalScale, setGlobalScale] = useState(DEFAULT_SETTINGS.globalScale);

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
                }
            } catch (error) {
                console.error('Failed to load global settings:', error);
            }
        };
        loadSettings();
    }, []);

    // 保存设置到 localStorage
    useEffect(() => {
        const saveSettings = () => {
            try {
                const settings = {
                    fontFamily,
                    customFont,
                    globalScale,
                };
                localStorage.setItem('globalSettings', JSON.stringify(settings));
            } catch (error) {
                console.error('Failed to save global settings:', error);
            }
        };
        saveSettings();
    }, [fontFamily, customFont, globalScale]);

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

    // 应用缩放到 CSS
    useEffect(() => {
        document.documentElement.style.setProperty('--global-scale', globalScale.toString());
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
    };

    // 导出设置
    const exportSettings = () => {
        const settings = {
            fontFamily,
            customFont,
            globalScale,
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
            return true;
        } catch (error) {
            console.error('Failed to import settings:', error);
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

        // 设置方法
        setFontFamily,
        setCustomFont,
        setGlobalScale,

        // 工具方法
        resetSettings,
        exportSettings,
        importSettings,
        getCurrentFontName,

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
