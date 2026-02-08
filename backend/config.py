from pydantic import Field, field_validator
try:
    from pydantic_settings import BaseSettings
    from pydantic import AliasChoices
except ImportError:  # 兼容旧版依赖
    from pydantic import BaseSettings
    AliasChoices = None  # type: ignore


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

    # ==================== OCR 配置 ====================
    # OCR 默认模式: auto（自动检测）/ always（始终启用）/ never（禁用）
    ocr_default_mode: str = Field(
        default="auto",
        validation_alias=AliasChoices("ocr_default_mode", "CHATPDF_OCR_DEFAULT_MODE"),
        description="OCR 默认模式: auto/always/never"
    )
    # OCR 图像转换 DPI（分辨率），范围 72-600
    ocr_dpi: int = Field(
        default=200,
        validation_alias=AliasChoices("ocr_dpi", "CHATPDF_OCR_DPI"),
    )
    # OCR 语言设置（Tesseract 语言代码）
    ocr_language: str = Field(
        default="chi_sim+eng",
        validation_alias=AliasChoices("ocr_language", "CHATPDF_OCR_LANGUAGE"),
    )
    # 首选 OCR 后端: auto / tesseract / paddleocr
    ocr_backend: str = Field(
        default="auto",
        validation_alias=AliasChoices("ocr_backend", "CHATPDF_OCR_BACKEND"),
    )
    # 页面质量阈值（0-100），低于此值的页面将触发 OCR
    ocr_quality_threshold: int = Field(
        default=60,
        validation_alias=AliasChoices("ocr_quality_threshold", "CHATPDF_OCR_QUALITY_THRESHOLD"),
    )

    # ==================== 在线 OCR 配置 ====================
    # Mistral OCR API Key
    mistral_ocr_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("mistral_ocr_api_key", "CHATPDF_MISTRAL_OCR_API_KEY"),
        description="Mistral OCR API Key"
    )
    # Mistral OCR API Base URL
    mistral_ocr_base_url: str = Field(
        default="https://api.mistral.ai",
        validation_alias=AliasChoices("mistral_ocr_base_url", "CHATPDF_MISTRAL_OCR_BASE_URL"),
        description="Mistral OCR API Base URL"
    )

    @field_validator("ocr_default_mode")
    @classmethod
    def validate_ocr_default_mode(cls, v: str) -> str:
        """校验 OCR 默认模式，仅接受 auto/always/never"""
        allowed = {"auto", "always", "never"}
        if v not in allowed:
            raise ValueError(
                f"ocr_default_mode 必须为 {allowed} 之一，当前值: {v!r}"
            )
        return v

    @field_validator("ocr_dpi")
    @classmethod
    def validate_ocr_dpi(cls, v: int) -> int:
        """校验 OCR DPI，范围 72-600"""
        if not (72 <= v <= 600):
            raise ValueError(
                f"ocr_dpi 必须在 72-600 范围内，当前值: {v}"
            )
        return v

    @field_validator("ocr_quality_threshold")
    @classmethod
    def validate_ocr_quality_threshold(cls, v: int) -> int:
        """校验 OCR 质量阈值，范围 0-100"""
        if not (0 <= v <= 100):
            raise ValueError(
                f"ocr_quality_threshold 必须在 0-100 范围内，当前值: {v}"
            )
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = AppSettings()
