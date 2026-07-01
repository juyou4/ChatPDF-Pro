"""GraphRAG 工具函数

包含 token 编码/解码、哈希计算、字符串处理、异步限流等通用工具。
"""

import asyncio
import html
import json
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union

import logging
import numpy as np

logger = logging.getLogger(__name__)

ENCODER = None


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4"):
    """使用 tiktoken 编码字符串为 token 列表"""
    global ENCODER
    if ENCODER is None:
        try:
            import tiktoken
            try:
                ENCODER = tiktoken.encoding_for_model(model_name)
            except Exception:
                ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except ImportError:
            # tiktoken 未安装时回退到字符级估算
            logger.warning("[GraphRAG] tiktoken 未安装，使用字符级 token 估算")
            return list(content)
    return ENCODER.encode(content)


def decode_tokens_by_tiktoken(tokens: list, model_name: str = "gpt-4"):
    """使用 tiktoken 解码 token 列表为字符串"""
    global ENCODER
    if ENCODER is None:
        try:
            import tiktoken
            try:
                ENCODER = tiktoken.encoding_for_model(model_name)
            except Exception:
                ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except ImportError:
            # 如果是字符列表（回退模式），直接拼接
            return "".join(str(t) for t in tokens)
    return ENCODER.decode(tokens)


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """按 token 数截断列表"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """计算内容的 MD5 哈希 ID"""
    return prefix + md5(content.encode()).hexdigest()


def compute_args_hash(*args) -> str:
    """计算参数的 MD5 哈希"""
    return md5(str(args).encode()).hexdigest()


def write_json(json_obj, file_name):
    """写入 JSON 文件"""
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    """加载 JSON 文件"""
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def pack_user_ass_to_openai_messages(*args: str):
    """将交替的 user/assistant 内容打包为 OpenAI 消息格式"""
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def is_float_regex(value):
    """判断字符串是否为浮点数"""
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """按多个分隔符拆分字符串"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def list_of_list_to_csv(data: list[list]) -> str:
    """将二维列表转为 CSV 格式字符串"""
    return "\n".join(
        [",\t".join([str(data_dd) for data_dd in data_d]) for data_d in data]
    )


def clean_str(input: Any) -> str:
    """清理字符串：去除 HTML 转义和控制字符"""
    if not isinstance(input, str):
        return input
    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """从字符串中提取 JSON 主体"""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    return None


def convert_response_to_json(response: str) -> dict:
    """将 LLM 响应解析为 JSON"""
    json_str = locate_json_string_body_from_string(response)
    if json_str is None:
        logger.warning(f"[GraphRAG] 无法从响应中提取 JSON: {response[:200]}")
        return {}
    try:
        data = json.loads(json_str)
        return data
    except Exception as e:
        logger.error(f"[GraphRAG] JSON 解析失败: {json_str[:200]}, error: {e}")
        return {}


def validate_embedding_result(embeddings, expected_dim: int, model_name: str = "") -> None:
    """校验 embedding 返回结果的维度和格式，在进入存储层之前报出清晰错误

    Raises:
        ValueError: 维度不匹配或格式异常
    """
    import numpy as np

    if embeddings is None:
        raise ValueError(f"[GraphRAG] Embedding 返回 None，模型 {model_name} 可能不可用")

    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"[GraphRAG] Embedding 返回类型异常: {type(embeddings)}，预期 np.ndarray")

    if embeddings.ndim != 2:
        raise ValueError(
            f"[GraphRAG] Embedding 维度数异常: ndim={embeddings.ndim}，预期 2 "
            f"(batch_size, embedding_dim)"
        )

    if embeddings.shape[0] == 0:
        raise ValueError("[GraphRAG] Embedding 返回空批次")

    actual_dim = int(embeddings.shape[1])
    if actual_dim != expected_dim:
        raise ValueError(
            f"[GraphRAG] Embedding 维度不匹配: 模型 {model_name} "
            f"预期 {expected_dim}，接口返回 {actual_dim}。"
            f"请检查 embedding 模型配置是否正确。"
        )

    # 检查 NaN/Inf
    if np.any(np.isnan(embeddings)):
        raise ValueError(f"[GraphRAG] Embedding 包含 NaN 值，模型 {model_name} 可能输出异常")
    if np.any(np.isinf(embeddings)):
        raise ValueError(f"[GraphRAG] Embedding 包含 Inf 值，模型 {model_name} 可能输出异常")


# ── 类型定义 ──

@dataclass
class EmbeddingFunc:
    """Embedding 函数包装器，携带维度和 token 限制元信息"""
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# ── 装饰器 ──

def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """异步函数并发限流装饰器"""

    def final_decro(func):
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            try:
                result = await func(*args, **kwargs)
            finally:
                __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """将普通 embedding 函数包装为 EmbeddingFunc"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro
