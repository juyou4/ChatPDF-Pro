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
            '/chat': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false
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
        sourcemap: true
    }
});
