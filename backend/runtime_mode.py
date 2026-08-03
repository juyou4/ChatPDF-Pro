"""
运行模式检测与全局配置

支持两种运行模式：
- server：本地部署模式（start.bat / Docker），支持本地 ML 模型
- desktop：桌面应用模式（Electron + PyInstaller），仅使用远程 API

优先级：
1. 显式环境变量 CHATPDF_MODE（最高优先级，方便调试）
2. sys.frozen 推断（PyInstaller 打包产物默认 desktop）
3. 兜底 server（源码运行默认 server）
"""

import ipaddress
import os
import sys
import logging
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class RuntimeConfig(BaseSettings):
    """运行时配置（从环境变量自动读取）"""

    # 运行模式：desktop | server（空字符串时自动推断）
    CHATPDF_MODE: str = ""

    # 用户数据目录。桌面端由 Electron 传入；源码运行也可显式指定，
    # 让代码工作树与用户数据目录彼此独立。
    CHATPDF_DATA_DIR: str = ""

    # 后端安全 token（桌面模式下由 Electron 生成并传入）
    CHATPDF_BACKEND_TOKEN: str = ""

    # Server 模式默认只绑定回环地址。需要局域网/公网部署时必须显式设置，
    # 并同时配置 CHATPDF_BACKEND_TOKEN 与 CHATPDF_CORS_ORIGINS。
    CHATPDF_HOST: str = ""

    # 逗号分隔的跨域白名单。留空时仅允许本地开发前端。
    CHATPDF_CORS_ORIGINS: str = ""

    # 监听端口（桌面模式下由 Electron 分配）
    CHATPDF_PORT: int = 8000

    # 日志级别
    CHATPDF_LOG_LEVEL: str = "INFO"

    class Config:
        env_prefix = ""  # 直接使用字段名作为环境变量名

    @property
    def is_desktop(self) -> bool:
        """判断是否为桌面模式"""
        if self.CHATPDF_MODE:
            return self.CHATPDF_MODE.lower() == "desktop"
        # PyInstaller 打包后 sys.frozen == True
        return getattr(sys, 'frozen', False)

    @property
    def is_server(self) -> bool:
        """判断是否为服务器模式"""
        return not self.is_desktop

    @property
    def mode_name(self) -> str:
        """返回当前模式名称"""
        return "desktop" if self.is_desktop else "server"

    @property
    def data_dir(self) -> str:
        """获取数据目录路径

        显式配置 CHATPDF_DATA_DIR 时，任何运行模式均使用该目录。
        否则桌面模式使用用户数据目录，源码运行使用项目根目录下的 data/。
        """
        if self.CHATPDF_DATA_DIR:
            return self.CHATPDF_DATA_DIR

        if self.is_desktop:
            # 桌面模式兜底：使用平台标准位置
            if sys.platform == "win32":
                base = os.environ.get("APPDATA", os.path.expanduser("~"))
            elif sys.platform == "darwin":
                base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
            else:
                base = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share"))
            return os.path.join(base, "ChatPDF")

        # 服务器模式：项目根目录/data
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(backend_dir)
        return os.path.join(project_dir, "data")

    @property
    def host(self) -> str:
        """监听地址：默认仅限本机，远程部署必须显式指定。"""
        if self.is_desktop:
            return "127.0.0.1"
        return (self.CHATPDF_HOST or "127.0.0.1").strip()

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        """判断监听地址是否只暴露给本机。"""
        normalized = (host or "").strip().strip("[]").lower()
        if normalized in {"localhost", "localhost.localdomain"}:
            return True
        try:
            return ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            return False

    @property
    def is_remote_host(self) -> bool:
        """是否显式监听在非回环地址。"""
        return not self._is_loopback_host(self.host)

    @property
    def cors_origins(self) -> list[str]:
        """返回明确的 CORS 白名单，绝不使用通配符。"""
        configured = [
            origin.strip().rstrip("/")
            for origin in self.CHATPDF_CORS_ORIGINS.split(",")
            if origin.strip()
        ]
        if configured:
            if "*" in configured:
                raise ValueError("CHATPDF_CORS_ORIGINS 不支持 '*'，请配置明确的受信任来源")
            return list(dict.fromkeys(configured))

        local_origins = [
            "http://127.0.0.1:3000",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]
        # 打包后的 Electron renderer 使用 file://，其 Origin 为 null。桌面
        # 后端始终要求随机 token，因此允许该 Origin 不会降低网络边界。
        if self.is_desktop:
            return [*local_origins, "null"]
        # 远程部署必须由管理员显式配置允许源；默认不发送 CORS 响应头。
        return [] if self.is_remote_host else local_origins

    def validate_startup_security(self) -> None:
        """在真正监听端口前验证运行模式的最低安全条件。"""
        if self.is_desktop and not self.CHATPDF_BACKEND_TOKEN:
            raise RuntimeError("桌面模式必须配置 CHATPDF_BACKEND_TOKEN")
        if self.is_remote_host and not self.CHATPDF_BACKEND_TOKEN:
            raise RuntimeError(
                "非回环地址监听必须配置 CHATPDF_BACKEND_TOKEN；"
                "如需远程前端访问，还必须配置 CHATPDF_CORS_ORIGINS"
            )
        if self.is_remote_host and not self.cors_origins:
            logger.warning("[RuntimeMode] 远程监听未配置 CHATPDF_CORS_ORIGINS，将拒绝所有跨域浏览器请求")

    @property
    def requires_token(self) -> bool:
        """是否需要 token 校验。桌面模式在配置缺失时也必须 fail closed。"""
        return self.is_desktop or bool(self.CHATPDF_BACKEND_TOKEN)


# 全局单例
runtime = RuntimeConfig()

# 启动时记录模式信息
logger.info(
    f"[RuntimeMode] 模式={runtime.mode_name}, "
    f"数据目录={runtime.data_dir}, "
    f"端口={runtime.CHATPDF_PORT}, "
    f"token校验={'启用' if runtime.requires_token else '关闭'}"
)
