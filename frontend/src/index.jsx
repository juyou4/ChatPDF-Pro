import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

import { ThreeLayerProvider } from './contexts';
import { GlobalSettingsProvider } from './contexts/GlobalSettingsContext';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <GlobalSettingsProvider>
            <ThreeLayerProvider>
                <App />
            </ThreeLayerProvider>
        </GlobalSettingsProvider>
    </React.StrictMode>
);
