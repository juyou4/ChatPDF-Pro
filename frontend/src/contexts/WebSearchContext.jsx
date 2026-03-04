import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';

// 联网搜索 Context —— 独立管理联网搜索相关状态
// 不污染 ChatParamsContext，避免不必要的重渲染

const WebSearchContext = createContext();

// 搜索引擎配置
export const WEB_SEARCH_PROVIDERS = [
    {
        id: 'duckduckgo',
        name: 'DuckDuckGo',
        description: '免费搜索，无需 API Key',
        requiresApiKey: false,
    },
    {
        id: 'tavily',
        name: 'Tavily',
        description: 'AI 原生搜索，免费 1000 次/月',
        requiresApiKey: true,
        apiKeyUrl: 'https://app.tavily.com/home',
    },
    {
        id: 'serper',
        name: 'Serper',
        description: 'Google 镜像搜索，新号 2500 次',
        requiresApiKey: true,
        apiKeyUrl: 'https://serper.dev/api-key',
    },
    {
        id: 'brave',
        name: 'Brave Search',
        description: '隐私优先，独立索引，不依赖 Google',
        requiresApiKey: true,
        apiKeyUrl: 'https://brave.com/search/api/',
    },
    {
        id: 'exa',
        name: 'Exa',
        description: 'AI 原生语义搜索，理解自然语言查询',
        requiresApiKey: true,
        apiKeyUrl: 'https://dashboard.exa.ai/api-keys',
    },
    {
        id: 'serpapi',
        name: 'SerpAPI',
        description: '多引擎 SERP（Google/Bing 等），每月 100 次免费',
        requiresApiKey: true,
        apiKeyUrl: 'https://serpapi.com/manage-api-key',
    },
    {
        id: 'google_cse',
        name: 'Google Custom Search',
        description: 'Google 官方搜索，每天 100 次免费',
        requiresApiKey: true,
        apiKeyUrl: 'https://programmablesearchengine.google.com/',
        apiKeyPlaceholder: 'API_KEY:CX_ID（冒号分隔）',
    },
    {
        id: 'firecrawl',
        name: 'Firecrawl',
        description: 'AI 搜索 + 内容提取，适合深度研究',
        requiresApiKey: true,
        apiKeyUrl: 'https://www.firecrawl.dev/app/api-keys',
    },
];

// 默认设置
export const WEB_SEARCH_DEFAULT_SETTINGS = {
    enableWebSearch: false,
    webSearchProvider: 'duckduckgo',
    webSearchApiKey: '',
};

export const WebSearchProvider = ({ children }) => {
    const [enableWebSearch, setEnableWebSearch] = useState(WEB_SEARCH_DEFAULT_SETTINGS.enableWebSearch);
    const [webSearchProvider, setWebSearchProvider] = useState(WEB_SEARCH_DEFAULT_SETTINGS.webSearchProvider);
    const [webSearchApiKey, setWebSearchApiKey] = useState(WEB_SEARCH_DEFAULT_SETTINGS.webSearchApiKey);

    // 防抖保存相关 ref
    const debounceTimerRef = useRef(null);
    const pendingSettingsRef = useRef(null);

    // 从 localStorage 加载设置
    useEffect(() => {
        try {
            const saved = localStorage.getItem('webSearchSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                if (settings.enableWebSearch !== undefined) setEnableWebSearch(settings.enableWebSearch);
                if (settings.webSearchProvider !== undefined) setWebSearchProvider(settings.webSearchProvider);
                if (settings.webSearchApiKey !== undefined) setWebSearchApiKey(settings.webSearchApiKey);
            }
        } catch (error) {
            console.error('加载联网搜索设置失败:', error);
        }
    }, []);

    // 防抖保存到 localStorage
    const flushSave = useCallback(() => {
        if (pendingSettingsRef.current !== null) {
            try {
                localStorage.setItem('webSearchSettings', JSON.stringify(pendingSettingsRef.current));
            } catch (error) {
                console.error('保存联网搜索设置失败:', error);
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

    // 监听设置变更，触发防抖保存
    useEffect(() => {
        const settings = {
            enableWebSearch,
            webSearchProvider,
            webSearchApiKey,
        };
        debouncedSave(settings);
    }, [enableWebSearch, webSearchProvider, webSearchApiKey, debouncedSave]);

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

    // 切换联网搜索开关
    const toggleWebSearch = useCallback(() => {
        setEnableWebSearch(prev => !prev);
    }, []);

    // 重置设置
    const resetWebSearch = useCallback(() => {
        setEnableWebSearch(WEB_SEARCH_DEFAULT_SETTINGS.enableWebSearch);
        setWebSearchProvider(WEB_SEARCH_DEFAULT_SETTINGS.webSearchProvider);
        setWebSearchApiKey(WEB_SEARCH_DEFAULT_SETTINGS.webSearchApiKey);
    }, []);

    // 获取当前 provider 配置
    const getCurrentProvider = useCallback(() => {
        return WEB_SEARCH_PROVIDERS.find(p => p.id === webSearchProvider) || WEB_SEARCH_PROVIDERS[0];
    }, [webSearchProvider]);

    const value = {
        // 状态
        enableWebSearch,
        webSearchProvider,
        webSearchApiKey,

        // 设置方法
        setEnableWebSearch,
        setWebSearchProvider,
        setWebSearchApiKey,

        // 工具方法
        toggleWebSearch,
        resetWebSearch,
        getCurrentProvider,
        flushSave,

        // 常量
        WEB_SEARCH_PROVIDERS,
        WEB_SEARCH_DEFAULT_SETTINGS,
    };

    return (
        <WebSearchContext.Provider value={value}>
            {children}
        </WebSearchContext.Provider>
    );
};

/**
 * 联网搜索 Hook
 */
export const useWebSearch = () => {
    const context = useContext(WebSearchContext);
    if (!context) {
        throw new Error('useWebSearch 必须在 WebSearchProvider 内部使用');
    }
    return context;
};

export default WebSearchContext;
