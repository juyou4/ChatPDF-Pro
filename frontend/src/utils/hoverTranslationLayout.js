/**
 * 悬浮译文面板的定位计算。
 *
 * 面板高度原先按固定值估算，但真实高度会随「要点 / 讲解」卡片和正文长度变化，
 * 靠近页面底部的段落于是把面板顶出可视区、底部被切掉。这里改用实测的外壳高度
 * （标题栏 + 内边距 + 要点卡片）与正文自然高度定位：先按可视带算出正文上限，
 * 再保证 top + 实际高度不越过可视带下沿。
 */

export const HOVER_PANEL_GAP = 16;
export const HOVER_PANEL_PAGE_PADDING = 18;
export const HOVER_PANEL_MIN_WIDTH = 260;
export const HOVER_PANEL_MAX_WIDTH = 380;
export const HOVER_PANEL_MIN_BODY_HEIGHT = 132;
export const HOVER_PANEL_MAX_BODY_HEIGHT = 360;
// 首帧还没量到真实尺寸时的兜底：标题栏加内边距约 92，正文按常见一屏估。
export const HOVER_PANEL_FALLBACK_CHROME_HEIGHT = 92;
export const HOVER_PANEL_FALLBACK_CONTENT_HEIGHT = 200;
// 可视带比这还窄时按这个高度算，否则面板会被压成看不清的一条缝。
const MIN_VISIBLE_BAND = 160;

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const numberOr = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
};

/**
 * 返回 `{ left, top, width, bodyMaxHeight, height }`，坐标是页面元素内的局部像素。
 * `height` 是面板落地后的实际高度，调用方可用它断言面板没有越界。
 */
export const resolveHoverTranslationLayout = ({
    bbox,
    scale = 1,
    pageWidthPts = 612,
    pageHeightPts = 792,
    pageRect = null,
    scrollRect = null,
    chromeHeight = 0,
    contentHeight = 0,
    measured = false,
}) => {
    if (!Array.isArray(bbox) || bbox.length < 4) return null;
    const [x0, y0, x1, y1] = bbox.slice(0, 4).map((value) => Number(value) * numberOr(scale, 1));
    if ([x0, y0, x1, y1].some((value) => !Number.isFinite(value))) return null;

    const gap = HOVER_PANEL_GAP;
    const pagePadding = HOVER_PANEL_PAGE_PADDING;
    const pageWidth = numberOr(pageWidthPts, 612) * numberOr(scale, 1);
    const pageHeight = numberOr(pageHeightPts, 792) * numberOr(scale, 1);
    const desiredWidth = Math.min(HOVER_PANEL_MAX_WIDTH, Math.max(300, pageWidth * 0.4));

    // 可视带 = 滚动容器与页面的交集，面板必须整块落在里面。
    const hasViewport = Boolean(pageRect && scrollRect);
    const bandTop = hasViewport
        ? Math.max(pagePadding, scrollRect.top - pageRect.top + pagePadding)
        : pagePadding;
    const bandBottom = Math.max(
        bandTop + MIN_VISIBLE_BAND,
        hasViewport
            ? Math.min(pageHeight - pagePadding, scrollRect.bottom - pageRect.top - pagePadding)
            : pageHeight - pagePadding,
    );

    const chrome = measured && chromeHeight > 0 ? chromeHeight : HOVER_PANEL_FALLBACK_CHROME_HEIGHT;
    const naturalBody = measured
        ? Math.max(0, numberOr(contentHeight, 0))
        : HOVER_PANEL_FALLBACK_CONTENT_HEIGHT;
    const bodyBudget = clamp(
        bandBottom - bandTop - chrome,
        HOVER_PANEL_MIN_BODY_HEIGHT,
        HOVER_PANEL_MAX_BODY_HEIGHT,
    );
    const panelHeight = chrome + Math.min(naturalBody, bodyBudget);
    const clampTop = (value) => clamp(value, bandTop, Math.max(bandTop, bandBottom - panelHeight));

    const leftGutter = hasViewport ? Math.max(0, pageRect.left - scrollRect.left - gap) : 0;
    const rightGutter = hasViewport ? Math.max(0, scrollRect.right - pageRect.right - gap) : 0;
    const blockCenterX = (x0 + x1) / 2;
    const preferRight = blockCenterX < pageWidth / 2;
    const canUseRightGutter = rightGutter >= HOVER_PANEL_MIN_WIDTH;
    const canUseLeftGutter = leftGutter >= HOVER_PANEL_MIN_WIDTH;

    let width = desiredWidth;
    let left = null;
    if (preferRight && canUseRightGutter) {
        width = Math.min(desiredWidth, rightGutter);
        left = pageWidth + gap;
    } else if (!preferRight && canUseLeftGutter) {
        width = Math.min(desiredWidth, leftGutter);
        left = -width - gap;
    } else if (canUseRightGutter || canUseLeftGutter) {
        const useRight = canUseRightGutter && (!canUseLeftGutter || rightGutter >= leftGutter);
        width = Math.min(desiredWidth, useRight ? rightGutter : leftGutter);
        left = useRight ? pageWidth + gap : -width - gap;
    }

    let top;
    if (left !== null) {
        top = clampTop(y0 - 8);
    } else {
        width = Math.min(desiredWidth, Math.max(HOVER_PANEL_MIN_WIDTH, pageWidth - pagePadding * 2));
        left = preferRight ? pageWidth - width - pagePadding : pagePadding;
        let preferredTop = y0 - 8;
        if (left < x1 && left + width > x0) {
            const belowTop = y1 + gap;
            const aboveTop = y0 - panelHeight - gap;
            if (belowTop + panelHeight <= bandBottom) {
                preferredTop = belowTop;
            } else if (aboveTop >= bandTop) {
                preferredTop = aboveTop;
            } else {
                // 上下都塞不下时贴住离段落更远的一侧，尽量少压住正文。
                preferredTop = Math.abs(y0 - bandTop) > Math.abs(bandBottom - y1)
                    ? bandTop
                    : bandBottom - panelHeight;
            }
        }
        top = clampTop(preferredTop);
    }

    // top 定死后正文再按剩余空间收一次，宁可内部滚动也不让底部被视口切掉。
    const bodyMaxHeight = clamp(bandBottom - top - chrome, HOVER_PANEL_MIN_BODY_HEIGHT, bodyBudget);

    return {
        left,
        top,
        width,
        bodyMaxHeight,
        height: chrome + Math.min(naturalBody, bodyMaxHeight),
    };
};
