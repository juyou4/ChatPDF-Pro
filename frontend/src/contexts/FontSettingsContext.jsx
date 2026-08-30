import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';

// 字体设置 Context —— 仅管理字体和缩放相关设置
// 从 GlobalSettingsContext 中拆分出来，实现细粒度订阅，
// 避免字体变更触发对话参数消费者的重渲染（需求 2.1, 2.2）

const FontSettingsContext = createContext();

const CJK_SANS_FALLBACK = '"Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif';
const CJK_SERIF_FALLBACK = '"Songti SC", "STSong", "SimSun", "Noto Serif CJK SC", serif';
const CJK_KAI_FALLBACK = '"KaiTi", "STKaiti", "Kaiti SC", serif';

// 预设字体列表。除 Google Fonts 外，中文字体优先使用设备已安装的字体；未安装时回退到系统中文字体。
export const PRESET_FONTS = [
    {
        id: 'source-han-sans',
        name: '思源黑体',
        value: `"ChatPDF Source Han Sans", "Source Han Sans SC", "Noto Sans SC", ${CJK_SANS_FALLBACK}`,
        description: '已内置 · 中文界面与长文阅读',
    },
    {
        id: 'alibaba-puhuiti',
        name: '阿里巴巴普惠体',
        value: `"Alibaba PuHuiTi 3.0", "Alibaba PuHuiTi 2.0", "Alibaba PuHuiTi", ${CJK_SANS_FALLBACK}`,
        description: '系统字体 · 本机安装时使用',
    },
    {
        id: 'oppo-sans',
        name: 'OPPO Sans',
        value: `"ChatPDF OPPO Sans", "OPPO Sans 4.0", "OPPO Sans", ${CJK_SANS_FALLBACK}`,
        description: '官方 Webfont · 在线时自动加载',
    },
    {
        id: 'hanyi-wenhei',
        name: '汉仪文黑',
        value: `"HYWenHei", "HYWenHei 85W", "汉仪文黑", ${CJK_SANS_FALLBACK}`,
        description: '系统字体 · 商用前请确认授权',
    },
    {
        id: 'gwm-sans',
        name: '长城共享体',
        value: `"GWM Sans", "长城共享体", ${CJK_SANS_FALLBACK}`,
        description: '系统字体 · 本机安装时使用',
    },
    {
        id: 'lxgw-neo-xihei',
        name: '霞鹜新晰黑',
        value: `"ChatPDF LXGW Neo XiHei", "LXGW Neo XiHei", "霞鹜新晰黑", ${CJK_SANS_FALLBACK}`,
        description: '已内置 · 中文正文干净规整',
    },
    {
        id: 'sarasa-gothic',
        name: '更纱黑体',
        value: `"ChatPDF Sarasa UI SC", "Sarasa UI SC", "Sarasa Gothic SC", "更纱黑体 SC", ${CJK_SANS_FALLBACK}`,
        description: '已内置 · 中文与代码混排稳定',
    },
    {
        id: 'lxgw-wenkai',
        name: '霞鹜文楷',
        value: `"ChatPDF LXGW WenKai", "LXGW WenKai", "霞鹜文楷", ${CJK_KAI_FALLBACK}`,
        description: '已内置 · 长文与笔记的文楷风格',
    },
    {
        id: 'source-han-serif',
        name: '思源宋体',
        value: `"ChatPDF Source Han Serif", "Source Han Serif SC", "Noto Serif CJK SC", ${CJK_SERIF_FALLBACK}`,
        description: '已内置 · 长文与电子书衬线阅读',
    },
    {
        id: 'screen-zhensong',
        name: '屏显臻宋',
        value: `"ChatPDF Clear Han Serif", "Clear Han Serif", "Screen ZhenSong", "屏显臻宋", ${CJK_SERIF_FALLBACK}`,
        description: '已内置 · 电子书与屏显长文宋体',
    },
    {
        id: 'canger-wenkai',
        name: '仓耳文楷 04W04',
        value: `"CangEr WenKai 04W04", "仓耳文楷 04W04", "LXGW WenKai", ${CJK_KAI_FALLBACK}`,
        description: '系统字体 · 商用前请确认授权',
    },
    {
        id: '975-yuanti',
        name: '975 圆体',
        value: `"ChatPDF 975 Yuan", "LXGW 975 Yuan SC", "975MaruSC", "975圆体", "975 圆体", ${CJK_SANS_FALLBACK}`,
        description: '已内置 · 中文界面与阅读圆润柔和',
    },
    { id: 'inter', name: 'Inter', value: 'Inter, sans-serif', googleFont: 'Inter:wght@300;400;500;600;700', description: 'Google Font · 严谨几何感的无衬线体' },
    {
        id: 'outfit-plus',
        name: 'Outfit + Plus',
        value: '"Plus Jakarta Sans", "Noto Sans SC", sans-serif',
        headingValue: 'Outfit, "Noto Sans SC", sans-serif',
        bodyValue: '"Plus Jakarta Sans", "Noto Sans SC", sans-serif',
        googleFont: 'Outfit:wght@400;700&family=Plus+Jakarta+Sans:wght@400;700',
        description: 'Outfit 标题 · Plus Jakarta Sans 正文',
    },
    {
        id: 'cormorant-work',
        name: 'Cormorant + Work',
        value: '"Work Sans", "Noto Sans SC", sans-serif',
        headingValue: 'Cormorant, "Noto Sans SC", sans-serif',
        bodyValue: '"Work Sans", "Noto Sans SC", sans-serif',
        googleFont: 'Cormorant:wght@400;700&family=Work+Sans:wght@400;700',
        description: 'Cormorant 标题 · Work Sans 正文',
    },
    {
        id: 'fraunces-karla',
        name: 'Fraunces + Karla',
        value: 'Karla, "Noto Sans SC", sans-serif',
        headingValue: 'Fraunces, "Noto Sans SC", sans-serif',
        bodyValue: 'Karla, "Noto Sans SC", sans-serif',
        googleFont: 'Fraunces:wght@400;700&family=Karla:wght@400;700',
        description: 'Fraunces 标题 · Karla 正文',
    },
    { id: 'roboto', name: 'Roboto', value: 'Roboto, sans-serif', googleFont: 'Roboto:wght@300;400;500;700', description: 'Google Font · 中性稳妥的界面字体' },
    { id: 'noto-sans-sc', name: 'Noto Sans SC', value: '"Noto Sans SC", sans-serif', googleFont: 'Noto+Sans+SC:wght@300;400;500;700', description: 'Google Font · CJK 多语言字重协调' },
    { id: 'poppins', name: 'Poppins', value: 'Poppins, sans-serif', googleFont: 'Poppins:wght@300;400;500;600;700', description: 'Google Font · 圆润几何感的标题字体' },
    { id: 'open-sans', name: 'Open Sans', value: '"Open Sans", sans-serif', googleFont: 'Open+Sans:wght@300;400;500;600;700', description: 'Google Font · 开口大、正文易读' },
    { id: 'lato', name: 'Lato', value: 'Lato, sans-serif', googleFont: 'Lato:wght@300;400;700', description: 'Google Font · 清爽干净的阅读体验' },
    { id: 'montserrat', name: 'Montserrat', value: 'Montserrat, sans-serif', googleFont: 'Montserrat:wght@300;400;500;600;700', description: 'Google Font · 宽阔有力的标题字体' },
];

// 字体相关默认设置
export const FONT_DEFAULT_SETTINGS = {
    fontFamily: 'source-han-sans',
    customFont: '',
    globalScale: 1.0,
    // 向后兼容旧版 globalSettings/fontSettings 中的 messageFont 字段
    messageFont: 'system',
};

const VALID_FONT_IDS = new Set([...PRESET_FONTS.map((font) => font.id), 'custom']);
const VALID_MESSAGE_FONTS = new Set(['system', 'serif']);

const normalizeFontSettings = (settings) => {
    if (!settings || typeof settings !== 'object' || Array.isArray(settings)) {
        return null;
    }

    const fontFamily = VALID_FONT_IDS.has(settings.fontFamily)
        ? settings.fontFamily
        : FONT_DEFAULT_SETTINGS.fontFamily;
    const customFont = typeof settings.customFont === 'string'
        ? settings.customFont
        : FONT_DEFAULT_SETTINGS.customFont;
    const parsedScale = Number(settings.globalScale);
    const globalScale = Number.isFinite(parsedScale) && parsedScale >= 0.5 && parsedScale <= 2
        ? parsedScale
        : FONT_DEFAULT_SETTINGS.globalScale;
    const messageFont = VALID_MESSAGE_FONTS.has(settings.messageFont)
        ? settings.messageFont
        : FONT_DEFAULT_SETTINGS.messageFont;

    return { fontFamily, customFont, globalScale, messageFont };
};

const readStoredFontSettings = () => {
    if (typeof window === 'undefined' || !window.localStorage) {
        return FONT_DEFAULT_SETTINGS;
    }

    for (const storageKey of ['fontSettings', 'globalSettings']) {
        try {
            const raw = window.localStorage.getItem(storageKey);
            if (!raw) continue;

            const normalized = normalizeFontSettings(JSON.parse(raw));
            if (normalized) return normalized;
        } catch (error) {
            console.error(`加载字体设置失败 (${storageKey}):`, error);
        }
    }

    return FONT_DEFAULT_SETTINGS;
};

/**
 * 加载 Google Font
 * @param {string} fontSpec - 字体规格，如 'Inter:wght@300;400;500' 或纯字体名称
 */
const loadGoogleFont = (fontSpec) => {
    const existingLink = document.getElementById('google-fonts-global');

    if (!fontSpec) {
        existingLink?.remove();
        return;
    }

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
    // 在首次渲染前同步完成恢复，避免默认值的保存副作用覆盖用户已选字体。
    const [initialSettings] = useState(readStoredFontSettings);
    const [fontFamily, setFontFamily] = useState(initialSettings.fontFamily);
    const [customFont, setCustomFont] = useState(initialSettings.customFont);
    const [globalScale, setGlobalScale] = useState(initialSettings.globalScale);
    const [messageFont, setMessageFont] = useState(initialSettings.messageFont);

    // 防抖保存相关 ref
    const debounceTimerRef = useRef(null);
    const pendingSettingsRef = useRef(null);

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
        const settings = { fontFamily, customFont, globalScale, messageFont };
        debouncedSave(settings);
    }, [fontFamily, customFont, globalScale, messageFont, debouncedSave]);

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
        let bodyFontValue;
        let headingFontValue;

        if (fontFamily === 'custom' && customFont) {
            // 使用自定义字体
            bodyFontValue = `"${customFont}", sans-serif`;
            headingFontValue = bodyFontValue;
            loadGoogleFont(customFont);
        } else {
            // 使用预设字体
            const preset = PRESET_FONTS.find(f => f.id === fontFamily);
            if (preset) {
                bodyFontValue = preset.bodyValue || preset.value;
                headingFontValue = preset.headingValue || bodyFontValue;
                loadGoogleFont(preset.googleFont);
            } else {
                bodyFontValue = PRESET_FONTS.find((font) => font.id === FONT_DEFAULT_SETTINGS.fontFamily)?.value || PRESET_FONTS[0].value;
                headingFontValue = bodyFontValue;
            }
        }

        // 保留旧变量以兼容现有样式，同时提供标题/正文双字体变量。
        document.documentElement.style.setProperty('--global-font-family', bodyFontValue);
        document.documentElement.style.setProperty('--body-font', bodyFontValue);
        document.documentElement.style.setProperty('--heading-font', headingFontValue);
    }, [fontFamily, customFont]);

    // 应用缩放到 html 根元素
    useEffect(() => {
        // globalScale 作为字体缩放因子，1.0 = 16px 基准
        const baseFontSize = 16;
        const fontSize = baseFontSize * globalScale;
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
        return preset ? preset.name : PRESET_FONTS.find((font) => font.id === FONT_DEFAULT_SETTINGS.fontFamily)?.name || '思源黑体';
    }, [fontFamily, customFont]);

    // 重置字体设置
    const resetFontSettings = useCallback(() => {
        setFontFamily(FONT_DEFAULT_SETTINGS.fontFamily);
        setCustomFont(FONT_DEFAULT_SETTINGS.customFont);
        setGlobalScale(FONT_DEFAULT_SETTINGS.globalScale);
        setMessageFont(FONT_DEFAULT_SETTINGS.messageFont);
    }, []);

    const value = {
        // 状态
        fontFamily,
        customFont,
        globalScale,
        messageFont,

        // 设置方法
        setFontFamily,
        setCustomFont,
        setGlobalScale,
        setMessageFont,

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
