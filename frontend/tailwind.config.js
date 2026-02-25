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
                    50: '#f7f5ff',
                    100: '#efebff',
                    200: '#e0d8ff',
                    300: '#c9bdff',
                    400: '#af9fff',
                    500: '#9884fc',
                    600: '#846cf2',
                    700: '#6f55e0',
                    800: '#5942be',
                    900: '#493899',
                    950: '#2d2166',
                }
            }
        },
    },
    plugins: [],
}
