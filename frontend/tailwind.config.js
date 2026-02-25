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
                    50: '#f8f7ff',
                    100: '#f0eeff',
                    200: '#e1ddff',
                    300: '#cac3ff',
                    400: '#b0a3ff',
                    500: '#9b8ef0',
                    600: '#8e7ee3',
                    700: '#7c6cd0',
                    800: '#6454b5',
                    900: '#524596',
                    950: '#342c63',
                }
            }
        },
    },
    plugins: [],
}
