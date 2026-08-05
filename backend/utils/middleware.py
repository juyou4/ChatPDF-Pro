from abc import ABC, abstractmethod
from typing import Any, Dict, List
import asyncio
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)
_ERROR_LOGGER_CACHE: dict[str, logging.Logger] = {}


class BaseMiddleware(ABC):
    """中间件基类，可在请求前后对payload/response做处理"""

    @abstractmethod
    async def before_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    @abstractmethod
    async def after_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return response


class LoggingMiddleware(BaseMiddleware):
    """简单日志中间件"""

    async def before_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["_ts"] = time.time()
        logger.debug(
            "[Middleware] -> Sending request to provider=%s model=%s",
            payload.get("provider"),
            payload.get("model"),
        )
        return payload

    async def after_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("[Middleware] <- Response received")
        return response


class ErrorCaptureMiddleware(BaseMiddleware):
    """捕获错误并写入运行时日志目录。"""

    def __init__(self, log_path: str = ""):
        self.log_path = str(self._resolve_log_path(log_path))
        self._error_logger = self._logger_for_path(self.log_path)

    @staticmethod
    def _resolve_log_path(log_path: str) -> Path:
        raw_path = str(log_path or "").strip()
        try:
            from runtime_mode import runtime
            base_dir = Path(runtime.data_dir)
        except Exception:
            base_dir = Path.cwd()

        if not raw_path:
            return base_dir / "logs" / "errors.log"

        path = Path(raw_path)
        if path.is_absolute():
            return path
        return base_dir / path

    @staticmethod
    def _logger_for_path(log_path: str) -> logging.Logger:
        path = Path(log_path)
        key = str(path.resolve())
        cached = _ERROR_LOGGER_CACHE.get(key)
        if cached is not None:
            return cached

        path.parent.mkdir(parents=True, exist_ok=True)
        error_logger = logging.getLogger(f"chatpdf.errors.{len(_ERROR_LOGGER_CACHE) + 1}")
        error_logger.setLevel(logging.INFO)
        error_logger.propagate = False
        handler = RotatingFileHandler(
            key,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        error_logger.addHandler(handler)
        _ERROR_LOGGER_CACHE[key] = error_logger
        return error_logger

    async def before_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    async def after_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(response, dict) and response.get("error"):
            try:
                self._error_logger.error("%s", response.get("error"))
            except Exception:
                # 静默失败以避免影响主流程
                pass
        return response


class DegradeOnErrorMiddleware(BaseMiddleware):
    """简单降级中间件：当上游报错时返回降级响应，并可携带备用内容"""

    def __init__(self, fallback_content: str = "服务繁忙，请稍后重试"):
        self.fallback_content = fallback_content

    async def before_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    async def after_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        # 如果 response 里有 error 字段，可包装降级内容
        if isinstance(response, dict) and response.get("error"):
            degraded = {
                "choices": [{
                    "message": {"content": self.fallback_content}
                }],
                "degraded": True,
                "answer_status": "degraded",
                "degrade_reason": "upstream_error",
            }
            # 保留调用身份和用量，供上层准确区分合成兜底文案与正常模型回答。
            for key in (
                "_used_provider",
                "_used_model",
                "_fallback_used",
                "usage",
                "_usage_meta",
            ):
                if key in response:
                    degraded[key] = response.get(key)
            return degraded
        return response


class FallbackMiddleware(BaseMiddleware):
    """当上游失败时，尝试备用模型/provider"""

    def __init__(self, fallback_provider: str | None = None, fallback_model: str | None = None):
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model

    async def before_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["_fallback_target"] = {
            "provider": self.fallback_provider,
            "model": self.fallback_model
        }
        return payload

    async def after_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(response, dict) and response.get("error"):
            # 在 response 中标记备用信息供上层读取
            response["_fallback"] = {
                "provider": self.fallback_provider,
                "model": self.fallback_model,
            }
        return response


class TimeoutMiddleware(BaseMiddleware):
    """在 payload 上标记超时，供客户端参考（实际超时由 httpx/客户端实现）"""

    def __init__(self, timeout: float):
        self.timeout = timeout

    async def before_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["_timeout"] = self.timeout
        return payload

    async def after_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return response


class RetryMiddleware(BaseMiddleware):
    """重试中间件，供调用方读取重试配置"""

    def __init__(self, retries: int = 2, delay: float = 0.5):
        self.retries = retries
        self.delay = delay

    async def before_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["_retry_cfg"] = {"retries": self.retries, "delay": self.delay}
        return payload

    async def after_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return response


async def apply_middlewares_before(payload: Dict[str, Any], middlewares: List[BaseMiddleware]) -> Dict[str, Any]:
    for mw in middlewares or []:
        payload = await mw.before_request(payload)
    return payload


async def apply_middlewares_after(response: Dict[str, Any], middlewares: List[BaseMiddleware]) -> Dict[str, Any]:
    for mw in middlewares or []:
        response = await mw.after_response(response)
    return response
