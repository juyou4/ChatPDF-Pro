"""API Key 池解析与随机选择

支持逗号分隔的多 Key 字符串，随机选择一个有效 Key。
"""
import random
from typing import List, Optional


def parse_api_key_pool(api_key_string: str) -> List[str]:
    """将逗号分隔的 API Key 字符串解析为有效 Key 列表

    按逗号分割输入字符串，对每个元素执行 trim 操作去除首尾空白，
    并过滤掉空字符串，返回有效 Key 的列表。

    Args:
        api_key_string: 逗号分隔的 Key 字符串，如 "key1, key2, key3"

    Returns:
        去除首尾空白、过滤空字符串后的有效 Key 列表
    """
    if not api_key_string:
        return []
    return [k.strip() for k in api_key_string.split(",") if k.strip()]


def select_api_key(api_key_string: str) -> Optional[str]:
    """从 API Key 池中随机选择一个有效 Key

    解析 API Key 池字符串，从有效 Key 中随机选择一个返回。
    单个 Key 时直接返回该 Key，空池返回 None。

    Args:
        api_key_string: 逗号分隔的 Key 字符串

    Returns:
        随机选择的 Key，池为空时返回 None
    """
    keys = parse_api_key_pool(api_key_string)
    if not keys:
        return None
    if len(keys) == 1:
        return keys[0]
    return random.choice(keys)
