import React, { useState } from 'react';
import { X, Globe, Check, ExternalLink, Eye, EyeOff, RotateCcw } from 'lucide-react';
import { useWebSearch, WEB_SEARCH_PROVIDERS } from '../contexts/WebSearchContext';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * 联网搜索设置面板
 * 
 * 类似 ChatSettings.jsx 风格的弹窗面板，包含：
 * - 搜索引擎选择（单选列表）
 * - API Key 输入框（选择需要 Key 的引擎时显示）
 * - 重置按钮
 */
const WebSearchSettings = ({ isOpen, onClose }) => {
    const {
        enableWebSearch,
        webSearchProvider,
        webSearchApiKey,
        setEnableWebSearch,
        setWebSearchProvider,
        setWebSearchApiKey,
        resetWebSearch,
    } = useWebSearch();

    const [showApiKey, setShowApiKey] = useState(false);

    const currentProvider = WEB_SEARCH_PROVIDERS.find(p => p.id === webSearchProvider) || WEB_SEARCH_PROVIDERS[0];
    const needsApiKey = currentProvider.requiresApiKey;

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
                    className="soft-panel rounded-2xl shadow-2xl max-w-md w-full max-h-[90vh] overflow-auto"
                    onClick={(e) => e.stopPropagation()}
                >
                    {/* 头部 */}
                    <div className="sticky top-0 bg-white/90 backdrop-blur-md border-b border-gray-200 px-6 py-4 flex items-center justify-between z-10">
                        <div className="flex items-center gap-3">
                            <div className="w-9 h-9 bg-blue-100 rounded-lg flex items-center justify-center">
                                <Globe className="w-5 h-5 text-blue-600" />
                            </div>
                            <div>
                                <h2 className="text-xl font-semibold text-gray-900">联网搜索设置</h2>
                                <p className="text-xs text-gray-500">选择搜索引擎并配置 API Key</p>
                            </div>
                        </div>
                        <button onClick={onClose} className="p-1.5 hover:bg-gray-100 rounded-md transition-colors">
                            <X className="w-5 h-5 text-gray-500" />
                        </button>
                    </div>

                    <div className="p-6 space-y-6">

                        {/* 联网搜索开关 */}
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-semibold text-gray-800">启用联网搜索</span>
                            </div>
                            <button
                                onClick={() => setEnableWebSearch(!enableWebSearch)}
                                className={`relative w-11 h-6 rounded-full transition-colors ${enableWebSearch ? 'bg-blue-500' : 'bg-gray-300'}`}
                            >
                                <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform ${enableWebSearch ? 'translate-x-5' : ''}`} />
                            </button>
                        </div>

                        <div className="border-t border-gray-100"></div>

                        {/* 搜索引擎选择 */}
                        <div className="space-y-3">
                            <span className="text-sm font-semibold text-gray-800">搜索引擎</span>
                            <div className="space-y-2">
                                {WEB_SEARCH_PROVIDERS.map((provider) => (
                                    <button
                                        key={provider.id}
                                        onClick={() => setWebSearchProvider(provider.id)}
                                        className={`w-full p-3 rounded-xl transition-all text-left relative ${
                                            webSearchProvider === provider.id
                                                ? 'bg-blue-50 ring-2 ring-blue-400'
                                                : 'bg-gray-50 hover:bg-gray-100'
                                        }`}
                                    >
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <div className="flex items-center gap-2">
                                                    <span className="font-medium text-gray-900 text-sm">{provider.name}</span>
                                                    {!provider.requiresApiKey && (
                                                        <span className="text-[10px] px-1.5 py-0.5 bg-green-100 text-green-700 rounded-full font-medium">免费</span>
                                                    )}
                                                </div>
                                                <p className="text-xs text-gray-500 mt-0.5">{provider.description}</p>
                                            </div>
                                            {webSearchProvider === provider.id && (
                                                <div className="bg-blue-500 rounded-full p-0.5 flex-shrink-0">
                                                    <Check className="w-3 h-3 text-white" />
                                                </div>
                                            )}
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* API Key 输入（仅当选择需要 Key 的引擎时显示） */}
                        {needsApiKey && (
                            <>
                                <div className="border-t border-gray-100"></div>
                                <div className="space-y-3">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm font-semibold text-gray-800">API Key</span>
                                        {currentProvider.apiKeyUrl && (
                                            <a
                                                href={currentProvider.apiKeyUrl}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-700 transition-colors"
                                            >
                                                获取 Key
                                                <ExternalLink className="w-3 h-3" />
                                            </a>
                                        )}
                                    </div>
                                    <div className="relative">
                                        <input
                                            type={showApiKey ? 'text' : 'password'}
                                            value={webSearchApiKey}
                                            onChange={(e) => setWebSearchApiKey(e.target.value)}
                                            placeholder={`输入 ${currentProvider.name} API Key`}
                                            className="w-full px-4 py-2.5 pr-10 bg-gray-50 border border-gray-200 rounded-xl outline-none text-sm focus:ring-2 focus:ring-blue-200 focus:border-blue-400 transition-all"
                                        />
                                        <button
                                            onClick={() => setShowApiKey(!showApiKey)}
                                            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                                        >
                                            {showApiKey
                                                ? <EyeOff className="w-4 h-4" />
                                                : <Eye className="w-4 h-4" />
                                            }
                                        </button>
                                    </div>
                                    {!webSearchApiKey && (
                                        <p className="text-xs text-amber-600">
                                            未配置 API Key，将回退到 DuckDuckGo 免费搜索
                                        </p>
                                    )}
                                </div>
                            </>
                        )}

                        <div className="border-t border-gray-100"></div>

                        {/* 重置按钮 */}
                        <button
                            onClick={() => { if (confirm('确定要重置联网搜索设置为默认值吗？')) resetWebSearch(); }}
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

export default WebSearchSettings;
