const brand = {
    50: '#fff8f5',
    100: '#fff0ea',
    200: '#ffdccf',
    300: '#ffc6b3',
    400: '#ffb092',
    500: '#ffa07a',
    600: '#b85f47',
    700: '#984a37',
    800: '#763729',
    900: '#52251c',
    950: '#32150f',
};

/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,jsx,ts,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // 品牌暖橙色阶（原紫色系整体旋转到 hue≈16°）。
                // violet/indigo 与 purple 指向同一套：历史代码里三个名字混用，
                // 统一在这里收口，避免逐文件改类名。
                purple: brand,
                violet: brand,
                indigo: brand,
            }
        },
    },
    plugins: [],
}
