const requireHttpUrl = (value) => {
  const parsed = new URL(String(value || '').trim());
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error('只支持 HTTP 或 HTTPS 地址');
  }
  return parsed.toString();
};

export const writePlainTextToClipboard = async (value, options = {}) => {
  const text = String(value || '');
  if (!text) throw new Error('没有可复制的内容');

  const clipboard = options.clipboard ?? globalThis.navigator?.clipboard;
  if (typeof clipboard?.writeText === 'function') {
    await clipboard.writeText(text);
    return 'clipboard';
  }

  const documentObject = options.document ?? globalThis.document;
  if (!documentObject?.body || typeof documentObject.execCommand !== 'function') {
    throw new Error('当前环境不支持剪贴板写入');
  }

  const textarea = documentObject.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  documentObject.body.appendChild(textarea);
  textarea.select();
  const copied = documentObject.execCommand('copy');
  textarea.remove();
  if (!copied) throw new Error('复制失败，请检查系统剪贴板权限');
  return 'legacy';
};

export const buildSelectionSearchUrl = ({ engine, customUrl, query }) => {
  const encodedQuery = encodeURIComponent(String(query || '').trim());
  if (!encodedQuery) throw new Error('没有可搜索的文字');

  const templates = {
    google: `https://www.google.com/search?q=${encodedQuery}`,
    bing: `https://www.bing.com/search?q=${encodedQuery}`,
    baidu: `https://www.baidu.com/s?wd=${encodedQuery}`,
    sogou: `https://www.sogou.com/web?query=${encodedQuery}`,
  };
  if (engine !== 'custom') return templates[engine] || templates.google;

  const rawTemplate = String(customUrl || '').trim();
  if (!rawTemplate) throw new Error('请先配置自定义搜索地址');
  const resolved = rawTemplate.includes('{query}')
    ? rawTemplate.replaceAll('{query}', encodedQuery)
    : `${rawTemplate}${rawTemplate.includes('?') ? '&' : '?'}q=${encodedQuery}`;
  return requireHttpUrl(resolved);
};

export const openExternalHttpUrl = async (value, options = {}) => {
  const url = requireHttpUrl(value);
  const desktopBridge = options.desktopBridge ?? globalThis.window?.chatpdfDesktop;
  if (typeof desktopBridge?.openExternal === 'function') {
    await desktopBridge.openExternal(url);
    return 'desktop';
  }

  const openWindow = options.openWindow ?? globalThis.window?.open;
  if (typeof openWindow !== 'function') throw new Error('当前环境无法打开外部链接');
  openWindow(url, '_blank', 'noopener,noreferrer');
  return 'browser';
};
