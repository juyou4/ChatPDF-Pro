import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
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
        rollupOptions: {
            output: {
                manualChunks: {
                    'vendor-react': ['react', 'react-dom'],
                    'vendor-motion': ['framer-motion'],
                    'vendor-markdown': [
                        'react-markdown',
                        'remark-gfm',
                        'remark-math',
                        'rehype-raw',
                        'rehype-highlight',
                        'rehype-katex',
                    ],
                    'vendor-mermaid': ['mermaid'],
                    'vendor-pdf': ['pdfjs-dist', 'react-pdf'],
                },
            },
        },
    }
});
