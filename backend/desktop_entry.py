"""
桌面应用入口脚本

由 Electron 主进程 spawn 调用，通过环境变量接收配置：
- CHATPDF_MODE=desktop
- CHATPDF_PORT=<port>
- CHATPDF_DATA_DIR=<path>
- CHATPDF_BACKEND_TOKEN=<token>
- CHATPDF_LOG_LEVEL=INFO|DEBUG

特性：
- 强制绑定 127.0.0.1（不暴露到局域网）
- 日志写入 {data_dir}/logs/，滚动 + 大小限制
- 优雅关闭处理（SIGTERM/SIGINT + Windows CTRL_C_EVENT）
"""

import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler

# 确保 backend 目录在 sys.path 中（PyInstaller 打包后可能需要）
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后，__file__ 指向临时目录
    _base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    if _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)
else:
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    if _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)

from runtime_mode import runtime


def setup_logging():
    """配置日志：滚动文件 + 控制台输出"""
    log_level = getattr(logging, runtime.CHATPDF_LOG_LEVEL.upper(), logging.INFO)

    # 确保日志目录存在
    log_dir = os.path.join(runtime.data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "chatpdf-backend.log")

    # 滚动日志：单文件 10MB，保留 5 个备份（总计 ≤ 60MB）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("desktop_entry")


def ensure_data_dirs():
    """确保数据目录结构存在"""
    dirs = [
        runtime.data_dir,
        os.path.join(runtime.data_dir, "logs"),
        os.path.join(runtime.data_dir, "docs"),
        os.path.join(runtime.data_dir, "vector_stores"),
        os.path.join(runtime.data_dir, "semantic_groups"),
        os.path.join(runtime.data_dir, "memory"),
        os.path.join(runtime.data_dir, "cache"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    logger = setup_logging()
    ensure_data_dirs()

    logger.info(
        f"ChatPDF Desktop Backend 启动: "
        f"mode={runtime.mode_name}, "
        f"host={runtime.host}, "
        f"port={runtime.CHATPDF_PORT}, "
        f"data_dir={runtime.data_dir}, "
        f"token={'已配置' if runtime.requires_token else '未配置'}"
    )

    # 优雅关闭处理
    def shutdown_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在关闭...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Windows CTRL_C_EVENT
    if sys.platform == "win32":
        try:
            signal.signal(signal.SIGBREAK, shutdown_handler)
        except (AttributeError, OSError):
            pass

    # 启动 uvicorn（直接传入 app 对象，PyInstaller 打包后字符串导入不可用）
    import uvicorn
    from app import app as application

    uvicorn.run(
        application,
        host=runtime.host,  # 桌面模式强制 127.0.0.1
        port=runtime.CHATPDF_PORT,
        log_level=runtime.CHATPDF_LOG_LEVEL.lower(),
        # 直接传对象时 reload 必须为 False
        reload=False,
    )


if __name__ == "__main__":
    main()
