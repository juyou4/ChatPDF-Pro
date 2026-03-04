"""URL 索引服务

使用 Jina Reader API 将网页 URL 转为 Markdown 文本，
然后复用现有的分块+索引流程将网页内容索引到向量库。

参考 kotaemon 的 WebReader（仅 44 行）。
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

JINA_READER_BASE = "https://r.jina.ai/"


async def fetch_url_content(
    url: str,
    timeout: float = 30.0,
) -> dict:
    """使用 Jina Reader API 抓取网页正文

    Args:
        url: 目标网页 URL
        timeout: 请求超时（秒）

    Returns:
        {"title": str, "content": str, "url": str}

    Raises:
        ValueError: URL 无效或抓取失败
    """
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError(f"无效的 URL: {url}")

    reader_url = f"{JINA_READER_BASE}{url}"

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(
                reader_url,
                headers={
                    "Accept": "text/plain",
                    "User-Agent": "ChatPDF-URLLoader/1.0",
                },
            )
            resp.raise_for_status()
            content = resp.text.strip()

        if not content:
            raise ValueError(f"Jina Reader 返回空内容: {url}")

        # Jina Reader 返回的第一行通常是标题
        lines = content.split("\n", 1)
        title = lines[0].strip().lstrip("# ") if lines else url
        body = lines[1].strip() if len(lines) > 1 else content

        logger.info(f"[URLLoader] 抓取成功: {url}, 标题={title[:50]}, 内容长度={len(body)}")
        return {
            "title": title,
            "content": body,
            "url": url,
        }

    except httpx.HTTPStatusError as e:
        raise ValueError(f"Jina Reader 请求失败 (HTTP {e.response.status_code}): {url}")
    except httpx.TimeoutException:
        raise ValueError(f"Jina Reader 请求超时: {url}")
    except Exception as e:
        raise ValueError(f"URL 抓取失败: {e}")
