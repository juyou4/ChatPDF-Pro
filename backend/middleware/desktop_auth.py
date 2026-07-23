"""
桌面模式安全中间件

在桌面模式下，Electron 启动后端时生成随机 BACKEND_TOKEN，
通过环境变量传给 Python。前端每次请求携带 X-ChatPDF-Token header，
后端中间件校验 token，不通过直接 401。

服务器模式在显式配置 CHATPDF_BACKEND_TOKEN 后也使用同一鉴权，
避免远程部署把全部 API 暴露为匿名接口。
"""

import logging
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# 不需要 token 校验的路径（健康检查必须放行，否则 Electron 无法检测后端就绪）
# Interactive documentation is not needed for normal desktop startup and
# should not disclose the complete API surface when a token is enabled.
EXEMPT_PATHS = {"/health", "/capabilities", "/favicon.ico"}


class DesktopAuthMiddleware(BaseHTTPMiddleware):
    """桌面模式请求鉴权中间件"""

    def __init__(self, app, runtime_config):
        super().__init__(app)
        self.runtime_config = runtime_config

    async def dispatch(self, request: Request, call_next):
        # 未启用 token 的本地 server 模式直接放行。
        if not self.runtime_config.requires_token:
            return await call_next(request)

        expected_token = self.runtime_config.CHATPDF_BACKEND_TOKEN
        # Electron 正常启动链会始终传入 token。若环境损坏或手工以 desktop
        # 模式启动，不能因空字符串比较而意外放行所有请求。
        if not expected_token:
            logger.error("[DesktopAuth] token 校验已启用但 CHATPDF_BACKEND_TOKEN 缺失")
            return JSONResponse(
                status_code=503,
                content={"detail": "Backend security token is not configured"},
            )

        # This middleware is outermost, while CORSMiddleware is responsible
        # for validating the origin/method/header combination. A browser CORS
        # preflight intentionally carries no application token, so let the
        # inner middleware answer it. The following actual request still goes
        # through the token check below.
        if (
            request.method == "OPTIONS"
            and request.headers.get("origin")
            and request.headers.get("access-control-request-method")
        ):
            return await call_next(request)

        # 豁免路径
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # 校验 token
        # 某些客户端/中间层重复设置同名 header 时，Starlette 可能读到 "token1, token2" 形式，
        # 这里取第一个值进行兼容处理，避免误判未授权。
        raw_token = request.headers.get("X-ChatPDF-Token", "")
        token = raw_token.split(",")[0].strip() if raw_token else ""
        if not token or not secrets.compare_digest(token, expected_token):
            logger.warning(
                f"[DesktopAuth] 未授权请求: {request.method} {request.url.path} "
                f"(来源: {request.client.host if request.client else 'unknown'})"
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: invalid or missing X-ChatPDF-Token"}
            )

        return await call_next(request)
