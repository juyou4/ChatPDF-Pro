import './config/desktop'; // 桌面模式适配（必须在最顶部，拦截 fetch/XHR）
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

import { ThreeLayerProvider } from './contexts';
import { GlobalSettingsProvider } from './contexts/GlobalSettingsContext';
import { WebSearchProvider } from './contexts/WebSearchContext';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <WebSearchProvider>
            <GlobalSettingsProvider>
                <ThreeLayerProvider>
                    <App />
                </ThreeLayerProvider>
            </GlobalSettingsProvider>
        </WebSearchProvider>
    </React.StrictMode>
);
