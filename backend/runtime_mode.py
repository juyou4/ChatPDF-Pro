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

import os
import sys
import logging
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class RuntimeConfig(BaseSettings):
    """运行时配置（从环境变量自动读取）"""

    # 运行模式：desktop | server（空字符串时自动推断）
    CHATPDF_MODE: str = ""

    # Electron 传入的用户数据目录（桌面模式必需）
    CHATPDF_DATA_DIR: str = ""

    # 后端安全 token（桌面模式下由 Electron 生成并传入）
    CHATPDF_BACKEND_TOKEN: str = ""

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

        桌面模式：使用 Electron 传入的 CHATPDF_DATA_DIR
        服务器模式：使用项目根目录下的 data/
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
        """监听地址：桌面模式强制 127.0.0.1"""
        if self.is_desktop:
            return "127.0.0.1"
        return "0.0.0.0"

    @property
    def requires_token(self) -> bool:
        """是否需要 token 校验"""
        return self.is_desktop and bool(self.CHATPDF_BACKEND_TOKEN)


# 全局单例
runtime = RuntimeConfig()

# 启动时记录模式信息
logger.info(
    f"[RuntimeMode] 模式={runtime.mode_name}, "
    f"数据目录={runtime.data_dir}, "
    f"端口={runtime.CHATPDF_PORT}, "
    f"token校验={'启用' if runtime.requires_token else '关闭'}"
)
