"""
桌面模式安全中间件

在桌面模式下，Electron 启动后端时生成随机 BACKEND_TOKEN，
通过环境变量传给 Python。前端每次请求携带 X-ChatPDF-Token header，
后端中间件校验 token，不通过直接 401。

服务器模式下此中间件自动跳过（不影响现有部署）。
"""

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# 不需要 token 校验的路径（健康检查必须放行，否则 Electron 无法检测后端就绪）
EXEMPT_PATHS = {"/health", "/capabilities", "/docs", "/openapi.json", "/favicon.ico"}


class DesktopAuthMiddleware(BaseHTTPMiddleware):
    """桌面模式请求鉴权中间件"""

    def __init__(self, app, runtime_config):
        super().__init__(app)
        self.runtime_config = runtime_config

    async def dispatch(self, request: Request, call_next):
        # 服务器模式：直接放行
        if not self.runtime_config.requires_token:
            return await call_next(request)

        # 豁免路径
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # 校验 token
        # 某些客户端/中间层重复设置同名 header 时，Starlette 可能读到 "token1, token2" 形式，
        # 这里取第一个值进行兼容处理，避免误判未授权。
        raw_token = request.headers.get("X-ChatPDF-Token", "")
        token = raw_token.split(",")[0].strip() if raw_token else ""
        if token != self.runtime_config.CHATPDF_BACKEND_TOKEN:
            logger.warning(
                f"[DesktopAuth] 未授权请求: {request.method} {request.url.path} "
                f"(来源: {request.client.host if request.client else 'unknown'})"
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: invalid or missing X-ChatPDF-Token"}
            )

        return await call_next(request)
