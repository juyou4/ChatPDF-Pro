/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,jsx,ts,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                purple: {
                    50: '#f3f0fc',
                    100: '#e7e2f9',
                    200: '#cfc6f4',
                    300: '#b7a9ee',
                    400: '#9f8de9',
                    500: '#8871e4',
                    600: '#6c5ab6',
                    700: '#514388',
                    800: '#362d5b',
                    900: '#1b162d',
                    950: '#0d0b16',
                }
            }
        },
    },
    plugins: [],
}
