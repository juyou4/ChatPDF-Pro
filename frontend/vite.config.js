import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

function manualChunks(id) {
    if (!id.includes('node_modules')) return undefined;

    const normalizedId = id.replaceAll('\\', '/');

    if (normalizedId.includes('/node_modules/pdfjs-dist/') || normalizedId.includes('/node_modules/react-pdf/')) {
        return 'vendor-pdf';
    }
    if (normalizedId.includes('/node_modules/rehype-mathjax/') || normalizedId.includes('/node_modules/mathjax-full/')) {
        return 'vendor-mathjax';
    }
    if (
        normalizedId.includes('/node_modules/react-markdown/') ||
        normalizedId.includes('/node_modules/remark-gfm/') ||
        normalizedId.includes('/node_modules/remark-math/') ||
        normalizedId.includes('/node_modules/rehype-raw/') ||
        normalizedId.includes('/node_modules/rehype-highlight/') ||
        normalizedId.includes('/node_modules/rehype-katex/')
    ) {
        return 'vendor-markdown';
    }
    if (
        normalizedId.includes('/node_modules/@codemirror/') ||
        normalizedId.includes('/node_modules/@lezer/')
    ) {
        return 'vendor-editor';
    }
    if (normalizedId.includes('/node_modules/framer-motion/')) return 'vendor-motion';
    if (
        normalizedId.includes('/node_modules/react/') ||
        normalizedId.includes('/node_modules/react-dom/') ||
        normalizedId.includes('/node_modules/scheduler/')
    ) {
        return 'vendor-react';
    }

    return undefined;
}

function isKnownMathjaxVersionEvalWarning(warning) {
    const id = warning.id?.replaceAll('\\', '/') || '';
    return warning.code === 'EVAL' && id.endsWith('/node_modules/mathjax-full/js/components/version.js');
}

function onRollupWarning(warning, warn) {
    if (isKnownMathjaxVersionEvalWarning(warning)) return;
    warn(warning);
}

export default defineConfig({
    base: './',
    plugins: [react()],
    server: {
        port: 3000,
        proxy: {
            '/upload': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/document': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/documents': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/models': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/embedding_models': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/health': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/api': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/chat': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false,
                // Disable proxy buffering so SSE chunks reach the browser immediately
                configure: (proxy) => {
                    proxy.on('proxyRes', (proxyRes) => {
                        // Flush each chunk to the browser right away (no Node buffering)
                        proxyRes.on('data', () => {});
                    });
                },
            },
            '/uploads': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/summary': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/storage_info': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            },
            '/capabilities': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
            }
        }
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src')
        }
    },
    build: {
        outDir: 'build',
        sourcemap: false,
        // MathJax 是用户选择 MathJax 渲染时才加载的按需块，当前约 1.8MB。
        chunkSizeWarningLimit: 2000,
        rollupOptions: {
            onwarn: onRollupWarning,
            output: {
                manualChunks,
            },
        },
    }
});
