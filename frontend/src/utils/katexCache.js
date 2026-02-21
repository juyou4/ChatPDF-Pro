/**
 * KaTeX 公式渲染缓存
 *
 * 模块级别的 LRU 缓存，所有 StreamingMarkdown 实例共享。
 * 缓存已渲染的 KaTeX HTML 结果，相同公式表达式直接返回缓存，
 * 避免重复渲染开销。
 *
 * LRU 策略：利用 Map 的插入顺序特性，
 * 访问时先 delete 再 set，将条目移到末尾（最近使用）。
 * 淘汰时删除 Map 迭代器的第一个条目（最久未使用）。
 *
 * @module katexCache
 * @see 需求 6.2 - KaTeX 公式渲染缓存
 */

// 默认最大缓存条目数
const DEFAULT_MAX_SIZE = 200;

/**
 * LRU 缓存类
 * 基于 Map 的插入顺序实现 LRU 淘汰策略
 */
class LRUCache {
  /**
   * @param {number} maxSize - 最大缓存条目数
   */
  constructor(maxSize = DEFAULT_MAX_SIZE) {
    this._map = new Map();
    this._maxSize = maxSize;
  }

  /**
   * 获取缓存值，命中时将条目移到末尾（标记为最近使用）
   * @param {string} key - 缓存键（LaTeX 表达式）
   * @returns {string|undefined} 缓存的 HTML 或 undefined
   */
  get(key) {
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
   * 设置缓存值，超出容量时淘汰最久未使用的条目
   * @param {string} key - 缓存键（LaTeX 表达式）
   * @param {string} value - 渲染后的 HTML
   */
  set(key, value) {
    // 如果 key 已存在，先删除以更新顺序
    if (this._map.has(key)) {
      this._map.delete(key);
    }
    this._map.set(key, value);
    // 超出容量时淘汰最旧条目（Map 迭代器第一个）
    if (this._map.size > this._maxSize) {
      const oldestKey = this._map.keys().next().value;
      this._map.delete(oldestKey);
    }
  }

  /**
   * 检查缓存中是否存在指定键
   * 注意：此方法不更新 LRU 顺序
   * @param {string} key - 缓存键
   * @returns {boolean}
   */
  has(key) {
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

// 模块级别单例，所有 StreamingMarkdown 实例共享
const katexCache = new LRUCache(DEFAULT_MAX_SIZE);

export { LRUCache, katexCache, DEFAULT_MAX_SIZE };
export default katexCache;
