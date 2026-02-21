/**
 * PDF 页面 canvas 缓存
 *
 * 基于 LRU 策略缓存已渲染 PDF 页面的 canvas 数据（dataURL），
 * 在短时间内回到同一页面时直接使用缓存图像，避免重新渲染。
 *
 * 缓存键格式：`${pageNumber}_${scale}`
 * 最大缓存 10 页，LRU 淘汰最久未访问的页面。
 *
 * 复用 katexCache 中的 LRU 实现模式（Map 插入顺序）。
 *
 * @module pdfPageCache
 * @see 需求 10.3 - PDF 页面 canvas 缓存
 */

// 默认最大缓存页数
const DEFAULT_MAX_PAGES = 10;

/**
 * PDF 页面缓存类
 * 基于 Map 的插入顺序实现 LRU 淘汰策略
 */
class PdfPageCache {
  /**
   * @param {number} maxSize - 最大缓存页数
   */
  constructor(maxSize = DEFAULT_MAX_PAGES) {
    this._map = new Map();
    this._maxSize = maxSize;
  }

  /**
   * 生成缓存键
   * @param {number} pageNumber - 页码
   * @param {number} scale - 缩放比例
   * @returns {string} 缓存键
   */
  static makeKey(pageNumber, scale) {
    return `${pageNumber}_${scale}`;
  }

  /**
   * 获取缓存的 canvas 数据，命中时将条目移到末尾（标记为最近使用）
   * @param {number} pageNumber - 页码
   * @param {number} scale - 缩放比例
   * @returns {string|undefined} 缓存的 dataURL 或 undefined
   */
  get(pageNumber, scale) {
    const key = PdfPageCache.makeKey(pageNumber, scale);
    if (!this._map.has(key)) {
      return undefined;
    }
    // 移到末尾：先删除再重新插入
    const value = this._map.get(key);
    this._map.delete(key);
    this._map.set(key, value);
    return value;
  }

  /**
   * 缓存页面的 canvas 数据，超出容量时淘汰最久未使用的条目
   * @param {number} pageNumber - 页码
   * @param {number} scale - 缩放比例
   * @param {string} dataURL - canvas.toDataURL() 的结果
   */
  set(pageNumber, scale, dataURL) {
    const key = PdfPageCache.makeKey(pageNumber, scale);
    // 如果 key 已存在，先删除以更新顺序
    if (this._map.has(key)) {
      this._map.delete(key);
    }
    this._map.set(key, dataURL);
    // 超出容量时淘汰最旧条目（Map 迭代器第一个）
    if (this._map.size > this._maxSize) {
      const oldestKey = this._map.keys().next().value;
      this._map.delete(oldestKey);
    }
  }

  /**
   * 检查缓存中是否存在指定页面
   * 注意：此方法不更新 LRU 顺序
   * @param {number} pageNumber - 页码
   * @param {number} scale - 缩放比例
   * @returns {boolean}
   */
  has(pageNumber, scale) {
    const key = PdfPageCache.makeKey(pageNumber, scale);
    return this._map.has(key);
  }

  /** 当前缓存条目数 */
  get size() {
    return this._map.size;
  }

  /** 最大缓存容量 */
  get maxSize() {
    return this._maxSize;
  }

  /** 清空缓存 */
  clear() {
    this._map.clear();
  }
}

// 模块级别单例，所有 PDFViewer 实例共享
const pdfPageCache = new PdfPageCache(DEFAULT_MAX_PAGES);

export { PdfPageCache, pdfPageCache, DEFAULT_MAX_PAGES };
export default pdfPageCache;
