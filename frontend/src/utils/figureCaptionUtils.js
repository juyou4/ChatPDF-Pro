// 悬停在 figure/table 主体上时，把悬浮层内容重定向到该图对应的图注（caption）块。
// 配对优先级与后端 block index 的绑定字段保持一致：
//   1. body.caption_block_id 直接指向 caption；
//   2. caption 的 linked_content_id(s) / caption_of / owner_block_id / figure_id 反向指向 body；
//   3. 同页空间兜底：caption 在图下方（表优先上方）、水平重叠足够、距离最近，且中间没有别的图表挡着。
// 找不到可用图注时返回 null，由调用方回退到原有 figure/table 行为。

const GAP_TOLERANCE = 12;
const MAX_CAPTION_GAP = 130;
const MIN_HORIZONTAL_OVERLAP = 0.3;

const normalizeBBox = (bbox) => {
    if (!Array.isArray(bbox) || bbox.length < 4) return null;
    const nums = bbox.slice(0, 4).map((value) => Number(value));
    if (nums.some((value) => !Number.isFinite(value))) return null;
    const [x0, y0, x1, y1] = nums;
    if (x1 <= x0 || y1 <= y0) return null;
    return [x0, y0, x1, y1];
};

export const isVisualBlock = (block) => block?.type === 'figure' || block?.type === 'table';

const hasCaptionText = (block) => Boolean(String(block?.text || '').trim());

// 图/表脚注（"* p<0.05"、数据来源说明）在 block index 里也归为 caption，
// 但它不是用户想看的那条 "Figure N: ..."，不参与配对。
const isFootnoteCaption = (block) => /footnote/i.test(String(block?.mineru_type || ''));

const captionLinkedBodyIds = (caption) => {
    const ids = new Set();
    const single = [caption?.linked_content_id, caption?.caption_of, caption?.owner_block_id];
    for (const value of single) {
        const id = String(value || '').trim();
        if (id) ids.add(id);
    }
    if (Array.isArray(caption?.linked_content_ids)) {
        for (const value of caption.linked_content_ids) {
            const id = String(value || '').trim();
            if (id) ids.add(id);
        }
    }
    return ids;
};

const horizontalOverlapRatio = (a, b) => {
    const overlap = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const minWidth = Math.max(1, Math.min(a[2] - a[0], b[2] - b[0]));
    return overlap / minWidth;
};

const verticalGapBetween = (bodyBBox, captionBBox) => {
    const aboveGap = bodyBBox[1] - captionBBox[3];
    const belowGap = captionBBox[1] - bodyBBox[3];
    const sides = [];
    if (aboveGap >= -GAP_TOLERANCE && aboveGap <= MAX_CAPTION_GAP) sides.push({ gap: Math.max(0, aboveGap), side: 'above' });
    if (belowGap >= -GAP_TOLERANCE && belowGap <= MAX_CAPTION_GAP) sides.push({ gap: Math.max(0, belowGap), side: 'below' });
    if (sides.length === 0) return null;
    return sides.reduce((best, item) => (item.gap < best.gap ? item : best));
};

// body 与 caption 的垂直间隙里若插着别的 figure/table，说明这条 caption 属于那张图。
const isBlockedBetween = (bodyBBox, captionBBox, side, blockerBBoxes) => {
    const bandTop = side === 'below' ? bodyBBox[3] : captionBBox[3];
    const bandBottom = side === 'below' ? captionBBox[1] : bodyBBox[1];
    if (bandBottom - bandTop <= 2) return false;
    return blockerBBoxes.some((blocker) => {
        const intersect = Math.min(bandBottom, blocker[3]) - Math.max(bandTop, blocker[1]);
        if (intersect <= 2) return false;
        return horizontalOverlapRatio(blocker, captionBBox) >= MIN_HORIZONTAL_OVERLAP
            || horizontalOverlapRatio(blocker, bodyBBox) >= MIN_HORIZONTAL_OVERLAP;
    });
};

const nearestCaption = (bodyBBox, captions) => {
    if (!bodyBBox) return captions[0] || null;
    let best = null;
    for (const caption of captions) {
        const capBBox = normalizeBBox(caption.bbox);
        if (!capBBox) continue;
        const placement = verticalGapBetween(bodyBBox, capBBox);
        const gap = placement ? placement.gap : Number.MAX_SAFE_INTEGER;
        if (!best || gap < best.gap) best = { gap, caption };
    }
    return best?.caption || captions[0] || null;
};

const findSpatialCaption = (block, captions, blocks) => {
    const bodyBBox = normalizeBBox(block.bbox);
    if (!bodyBBox) return null;
    const blockId = String(block.block_id || '').trim();
    const blockerBBoxes = blocks
        .filter((item) => item !== block && isVisualBlock(item))
        .map((item) => normalizeBBox(item.bbox))
        .filter(Boolean);
    const claimedByOtherBody = new Set();
    for (const other of blocks) {
        if (other === block || !isVisualBlock(other)) continue;
        const claimedId = String(other.caption_block_id || '').trim();
        if (claimedId) claimedByOtherBody.add(claimedId);
    }
    const prefer = block.type === 'table' ? 'above' : 'below';
    let best = null;
    for (const caption of captions) {
        if (isFootnoteCaption(caption)) continue;
        if (claimedByOtherBody.has(String(caption.block_id || '').trim())) continue;
        const linkedIds = captionLinkedBodyIds(caption);
        if (linkedIds.size > 0 && !(blockId && linkedIds.has(blockId))) continue;
        const capBBox = normalizeBBox(caption.bbox);
        if (!capBBox) continue;
        const overlapRatio = horizontalOverlapRatio(bodyBBox, capBBox);
        if (overlapRatio < MIN_HORIZONTAL_OVERLAP) continue;
        const placement = verticalGapBetween(bodyBBox, capBBox);
        if (!placement) continue;
        if (isBlockedBetween(bodyBBox, capBBox, placement.side, blockerBBoxes)) continue;
        const orientationPenalty = placement.side === prefer ? 0 : (prefer === 'below' ? 20 : 30);
        const score = placement.gap + orientationPenalty - overlapRatio * 20;
        if (!best || score < best.score) best = { score, caption };
    }
    return best?.caption || null;
};

/**
 * 在同页 blocks 里为 figure/table 主体块找对应的图注块。
 *
 * @param {object} block 悬停命中的 figure/table 块
 * @param {object[]} blocks 当前页全部块（含 caption）
 * @returns {object|null} 配对成功的 caption 块；找不到可靠配对时返回 null
 */
export const findAssociatedCaption = (block, blocks) => {
    if (!isVisualBlock(block) || !Array.isArray(blocks) || blocks.length === 0) return null;
    const blockId = String(block.block_id || '').trim();
    const figureId = String(block.figure_id || blockId || '').trim();
    const captions = blocks.filter((item) => item !== block && item?.type === 'caption' && hasCaptionText(item));
    if (captions.length === 0) return null;

    const explicitId = String(block.caption_block_id || '').trim();
    if (explicitId) {
        const explicit = captions.find((item) => String(item.block_id || '').trim() === explicitId);
        if (explicit) return explicit;
    }

    const linked = captions.filter((item) => {
        const ids = captionLinkedBodyIds(item);
        if (blockId && ids.has(blockId)) return true;
        const captionFigureId = String(item.figure_id || '').trim();
        return Boolean(captionFigureId && figureId && captionFigureId === figureId);
    });
    if (linked.length > 0) {
        const nonFootnote = linked.filter((item) => !isFootnoteCaption(item));
        const pool = nonFootnote.length > 0 ? nonFootnote : linked;
        if (pool.length === 1) return pool[0];
        return nearestCaption(normalizeBBox(block.bbox), pool);
    }

    return findSpatialCaption(block, captions, blocks);
};
