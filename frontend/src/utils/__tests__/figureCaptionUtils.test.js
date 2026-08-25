import { describe, it, expect } from 'vitest';
import { findAssociatedCaption, isVisualBlock } from '../figureCaptionUtils.js';

const figure = (overrides = {}) => ({
    block_id: 'p1_b1',
    type: 'figure',
    bbox: [100, 100, 400, 300],
    text: '',
    ...overrides,
});

const caption = (overrides = {}) => ({
    block_id: 'p1_b2',
    type: 'caption',
    bbox: [100, 310, 400, 340],
    text: 'Figure 2: The AdvRoad framework.',
    ...overrides,
});

describe('isVisualBlock', () => {
    it('只认 figure/table', () => {
        expect(isVisualBlock(figure())).toBe(true);
        expect(isVisualBlock(figure({ type: 'table' }))).toBe(true);
        expect(isVisualBlock(caption())).toBe(false);
        expect(isVisualBlock(null)).toBe(false);
    });
});

describe('findAssociatedCaption - 显式绑定', () => {
    it('body.caption_block_id 直接命中，优先于空间距离', () => {
        const body = figure({ caption_block_id: 'cap_far' });
        const near = caption({ block_id: 'cap_near', bbox: [100, 305, 400, 330] });
        const far = caption({ block_id: 'cap_far', bbox: [100, 380, 400, 410], text: 'Figure 2: the real one.' });
        expect(findAssociatedCaption(body, [body, near, far])).toBe(far);
    });

    it('caption.linked_content_id 反向指回 body 时命中', () => {
        const body = figure();
        const linked = caption({ linked_content_id: 'p1_b1' });
        expect(findAssociatedCaption(body, [body, linked])).toBe(linked);
    });

    it('caption_of / owner_block_id / figure_id 也可反向命中', () => {
        const body = figure({ figure_id: 'fig-2' });
        const byCaptionOf = caption({ block_id: 'c1', caption_of: 'p1_b1' });
        const byOwner = caption({ block_id: 'c2', owner_block_id: 'p1_b1' });
        const byFigureId = caption({ block_id: 'c3', figure_id: 'fig-2' });
        expect(findAssociatedCaption(figure(), [figure(), byCaptionOf])).toEqual(byCaptionOf);
        expect(findAssociatedCaption(figure(), [figure(), byOwner])).toEqual(byOwner);
        expect(findAssociatedCaption(body, [body, byFigureId])).toBe(byFigureId);
    });

    it('多条反向绑定时优先真图注而非图脚注', () => {
        const body = figure();
        const footnote = caption({
            block_id: 'fn',
            mineru_type: 'image_footnote',
            linked_content_id: 'p1_b1',
            bbox: [100, 345, 400, 360],
            text: '* p < 0.05',
        });
        const real = caption({ block_id: 'cap', linked_content_id: 'p1_b1' });
        expect(findAssociatedCaption(body, [body, footnote, real])).toBe(real);
    });
});

describe('findAssociatedCaption - 空间兜底', () => {
    it('无绑定字段时选图下方水平重叠且最近的 caption', () => {
        const body = figure();
        const below = caption({ block_id: 'below', bbox: [110, 312, 390, 336] });
        const farBelow = caption({ block_id: 'far', bbox: [110, 420, 390, 440], text: 'Figure 3: another.' });
        expect(findAssociatedCaption(body, [body, below, farBelow])).toBe(below);
    });

    it('table 优先选上方 caption', () => {
        const body = figure({ type: 'table', bbox: [100, 200, 400, 400] });
        const above = caption({ block_id: 'above', bbox: [100, 160, 400, 190], text: 'Table 1: results.' });
        const below = caption({ block_id: 'below', bbox: [100, 410, 400, 440], text: 'Figure 4: other.' });
        expect(findAssociatedCaption(body, [body, above, below])).toBe(above);
    });

    it('不误绑水平不重叠的邻栏图注', () => {
        // 双栏排版：左栏 figure，右栏另一张图的 caption。
        const body = figure({ bbox: [50, 100, 280, 300] });
        const otherColumn = caption({ block_id: 'right-col', bbox: [320, 310, 560, 340], text: 'Figure 9: unrelated.' });
        expect(findAssociatedCaption(body, [body, otherColumn])).toBeNull();
    });

    it('不抢已显式绑定到别的图的 caption', () => {
        const body = figure({ block_id: 'figA', bbox: [100, 100, 400, 300] });
        const otherBody = figure({ block_id: 'figB', bbox: [100, 400, 400, 600] });
        const claimed = caption({ block_id: 'capB', bbox: [100, 310, 400, 340], linked_content_id: 'figB' });
        expect(findAssociatedCaption(body, [body, otherBody, claimed])).toBeNull();
    });

    it('不抢被别的 body.caption_block_id 占用的 caption', () => {
        const body = figure({ block_id: 'figA', bbox: [100, 100, 400, 300] });
        const otherBody = figure({ block_id: 'figB', bbox: [100, 360, 400, 560], caption_block_id: 'capB' });
        const claimed = caption({ block_id: 'capB', bbox: [100, 305, 400, 335] });
        expect(findAssociatedCaption(body, [body, otherBody, claimed])).toBeNull();
    });

    it('中间隔着另一张图时不越过去配对', () => {
        const body = figure({ block_id: 'figA', bbox: [100, 100, 400, 200] });
        const blocker = figure({ block_id: 'figB', bbox: [100, 210, 400, 300] });
        const belowBlocker = caption({ block_id: 'capB', bbox: [100, 305, 400, 330] });
        expect(findAssociatedCaption(body, [body, blocker, belowBlocker])).toBeNull();
    });

    it('空间兜底不选图脚注', () => {
        const body = figure();
        const footnote = caption({
            block_id: 'fn',
            mineru_type: 'table_footnote',
            bbox: [100, 305, 400, 325],
            text: 'Source: authors.',
        });
        expect(findAssociatedCaption(body, [body, footnote])).toBeNull();
    });

    it('垂直距离超限时不配对', () => {
        const body = figure();
        const tooFar = caption({ block_id: 'far', bbox: [100, 480, 400, 510] });
        expect(findAssociatedCaption(body, [body, tooFar])).toBeNull();
    });
});

describe('findAssociatedCaption - 回退', () => {
    it('页面没有 caption 时返回 null', () => {
        const body = figure();
        const paragraph = { block_id: 'p1_b9', type: 'paragraph', bbox: [100, 320, 400, 360], text: 'Body text.' };
        expect(findAssociatedCaption(body, [body, paragraph])).toBeNull();
        expect(findAssociatedCaption(body, [body])).toBeNull();
    });

    it('caption 文本为空时视为不可用', () => {
        const body = figure({ caption_block_id: 'empty' });
        const empty = caption({ block_id: 'empty', text: '   ' });
        expect(findAssociatedCaption(body, [body, empty])).toBeNull();
    });

    it('非 figure/table 块不参与配对', () => {
        const heading = { block_id: 'h1', type: 'heading', bbox: [100, 100, 400, 130], text: '1. Intro' };
        expect(findAssociatedCaption(heading, [heading, caption()])).toBeNull();
    });
});
