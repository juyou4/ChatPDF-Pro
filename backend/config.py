from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:  # 兼容旧版依赖
    from pydantic import BaseSettings


class AppSettings(BaseSettings):
    """应用配置（Pydantic Settings，可覆盖 env）"""

    # 中间件控制
    enable_chat_logging: bool = Field(default=True, env="CHATPDF_CHAT_LOGGING")
    chat_retry_retries: int = Field(default=1, env="CHATPDF_CHAT_RETRY_RETRIES")
    chat_retry_delay: float = Field(default=0.5, env="CHATPDF_CHAT_RETRY_DELAY")

    # 检索链路
    enable_search_logging: bool = Field(default=True, env="CHATPDF_SEARCH_LOGGING")
    search_retry_retries: int = Field(default=1, env="CHATPDF_SEARCH_RETRY_RETRIES")
    search_retry_delay: float = Field(default=0.3, env="CHATPDF_SEARCH_RETRY_DELAY")

    # 降级
    enable_chat_degrade: bool = Field(default=False, env="CHATPDF_CHAT_DEGRADE")
    degrade_message: str = Field(default="服务繁忙，请稍后重试", env="CHATPDF_DEGRADE_MESSAGE")
    # 搜索降级
    enable_search_degrade: bool = Field(default=False, env="CHATPDF_SEARCH_DEGRADE")
    search_degrade_message: str = Field(default="搜索暂不可用，请稍后重试", env="CHATPDF_SEARCH_DEGRADE_MESSAGE")
    # 日志路径
    error_log_path: str = Field(default="logs/errors.log", env="CHATPDF_ERROR_LOG_PATH")
    # 超时/断路器
    chat_timeout: float = Field(default=120.0, env="CHATPDF_CHAT_TIMEOUT")
    search_timeout: float = Field(default=30.0, env="CHATPDF_SEARCH_TIMEOUT")

    # 备用模型/提供商（用于失败兜底）
    chat_fallback_provider: str | None = Field(default=None, env="CHATPDF_CHAT_FALLBACK_PROVIDER")
    chat_fallback_model: str | None = Field(default=None, env="CHATPDF_CHAT_FALLBACK_MODEL")
    search_fallback_provider: str | None = Field(default=None, env="CHATPDF_SEARCH_FALLBACK_PROVIDER")
    search_fallback_model: str | None = Field(default=None, env="CHATPDF_SEARCH_FALLBACK_MODEL")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = AppSettings()
