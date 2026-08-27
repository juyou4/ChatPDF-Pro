import { describe, it, expect } from 'vitest';
import { resolveHoverTranslationLayout } from '../hoverTranslationLayout';

// 视口 900px 高、页面 1400px 高且已向下滚动 300px 的场景。
// 可视带（页面局部坐标）= [318, 1182]。
const BAND_TOP = 318;
const BAND_BOTTOM = 1182;
const viewport = {
    pageRect: { top: -300, bottom: 1100, left: 0, right: 900 },
    scrollRect: { top: 0, bottom: 900, left: 0, right: 900 },
};
const baseArgs = {
    scale: 1,
    pageWidthPts: 800,
    pageHeightPts: 1400,
    ...viewport,
};

describe('resolveHoverTranslationLayout', () => {
    it('页面底部的段落：面板整块留在可视带内', () => {
        const layout = resolveHoverTranslationLayout({
            ...baseArgs,
            bbox: [60, 1150, 400, 1250],
            chromeHeight: 200,
            contentHeight: 500,
            measured: true,
        });

        expect(layout.top).toBeGreaterThanOrEqual(BAND_TOP);
        expect(layout.top + layout.height).toBeLessThanOrEqual(BAND_BOTTOM);
    });

    it('实测高度会把面板往上推，估算值则会漏算「要点」卡片而露出底部', () => {
        const args = { ...baseArgs, bbox: [60, 1150, 400, 1250] };
        const measured = resolveHoverTranslationLayout({
            ...args,
            chromeHeight: 200,
            contentHeight: 500,
            measured: true,
        });
        const estimated = resolveHoverTranslationLayout(args);

        expect(measured.top).toBeLessThan(estimated.top);
        // 估算位置配上真实高度就会捅出可视带 —— 这正是原先被切掉的那一块。
        expect(estimated.top + measured.height).toBeGreaterThan(BAND_BOTTOM);
    });

    it('可视带不够高时收正文滚动区，而不是让面板溢出', () => {
        const layout = resolveHoverTranslationLayout({
            ...baseArgs,
            scrollRect: { top: 0, bottom: 420, left: 0, right: 900 },
            bbox: [60, 1150, 400, 1250],
            chromeHeight: 200,
            contentHeight: 900,
            measured: true,
        });

        expect(layout.bodyMaxHeight).toBeLessThan(360);
        expect(layout.top + layout.height).toBeLessThanOrEqual(702);
    });

    it('走页面右侧留白时同样受可视带约束', () => {
        const layout = resolveHoverTranslationLayout({
            ...baseArgs,
            pageWidthPts: 600,
            pageRect: { top: -300, bottom: 1100, left: 300, right: 900 },
            scrollRect: { top: 0, bottom: 900, left: 0, right: 1400 },
            bbox: [40, 1150, 250, 1250],
            chromeHeight: 200,
            contentHeight: 500,
            measured: true,
        });

        expect(layout.left).toBe(616);
        expect(layout.top + layout.height).toBeLessThanOrEqual(BAND_BOTTOM);
    });

    it('段落落在页面任意高度都不越界', () => {
        for (let y = 0; y <= 1300; y += 50) {
            const layout = resolveHoverTranslationLayout({
                ...baseArgs,
                bbox: [60, y, 400, y + 100],
                chromeHeight: 240,
                contentHeight: 640,
                measured: true,
            });
            expect(layout.top).toBeGreaterThanOrEqual(BAND_TOP);
            expect(layout.top + layout.height).toBeLessThanOrEqual(BAND_BOTTOM);
        }
    });

    it('缺少滚动容器信息时退回整页可视带', () => {
        const layout = resolveHoverTranslationLayout({
            scale: 1,
            pageWidthPts: 800,
            pageHeightPts: 1400,
            bbox: [60, 1300, 400, 1380],
            chromeHeight: 200,
            contentHeight: 500,
            measured: true,
        });

        expect(layout.top).toBeGreaterThanOrEqual(18);
        expect(layout.top + layout.height).toBeLessThanOrEqual(1382);
    });

    it('bbox 非法时返回 null', () => {
        expect(resolveHoverTranslationLayout({ ...baseArgs, bbox: null })).toBeNull();
        expect(resolveHoverTranslationLayout({ ...baseArgs, bbox: [1, 2] })).toBeNull();
        expect(resolveHoverTranslationLayout({ ...baseArgs, bbox: [1, 2, 3, 'x'] })).toBeNull();
    });
});
