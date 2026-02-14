"""
通用异步并发工具

提供：
- run_with_concurrency: 异步并发池，限制同时执行的任务数
- retry_with_backoff: 指数退避重试，带随机抖动

参考 paper-burner-x 的 processWithConcurrencyLimit 和 retryWithBackoff 实现。
"""

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def run_with_concurrency(
    items: List[Any],
    handler: Callable[[Any], Awaitable[T]],
    limit: int = 10,
) -> List[T]:
    """异步并发池，限制同时执行的任务数

    使用 asyncio.Semaphore 控制并发上限，所有任务完成后返回结果列表。
    单个任务失败不影响其他任务（失败的任务返回 None）。

    Args:
        items: 待处理的项目列表
        handler: 异步处理函数，接受单个项目，返回结果
        limit: 最大并发数，默认 10

    Returns:
        结果列表，顺序与 items 对应
    """
    if not items:
        return []

    semaphore = asyncio.Semaphore(limit)
    results: List[T] = [None] * len(items)

    async def _run_one(index: int, item: Any):
        async with semaphore:
            try:
                results[index] = await handler(item)
            except Exception as e:
                logger.warning(
                    f"[Concurrency] 任务 {index} 失败: {e}"
                )
                results[index] = None

    tasks = [_run_one(i, item) for i, item in enumerate(items)]
    await asyncio.gather(*tasks)

    return results


async def retry_with_backoff(
    fn: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 0.6,
    max_delay: float = 5.0,
    **kwargs,
) -> T:
    """指数退避重试，带随机抖动

    每次失败后等待时间按指数增长（base_delay * 2^attempt），
    并加入 ±30% 的随机抖动以避免雷群效应。

    Args:
        fn: 要重试的异步函数
        *args: 传给 fn 的位置参数
        max_retries: 最大重试次数（不含首次调用），默认 3
        base_delay: 基础延迟秒数，默认 0.6
        max_delay: 最大延迟秒数，默认 5.0
        **kwargs: 传给 fn 的关键字参数

    Returns:
        fn 的返回值

    Raises:
        Exception: 所有重试均失败后抛出最后一次异常
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                # 指数退避 + 随机抖动
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = delay * random.uniform(-0.3, 0.3)
                actual_delay = max(0.1, delay + jitter)

                logger.info(
                    f"[Retry] 第 {attempt + 1}/{max_retries} 次重试，"
                    f"等待 {actual_delay:.2f}s，错误: {e}"
                )
                await asyncio.sleep(actual_delay)
            else:
                logger.error(
                    f"[Retry] 已达最大重试次数 {max_retries}，最终失败: {e}"
                )

    raise last_error
