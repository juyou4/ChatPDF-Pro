import logging
from typing import Any, Optional
from pydantic import Field, field_validator
try:
    from pydantic_settings import BaseSettings
    from pydantic import AliasChoices
except ImportError:  # 兼容旧版依赖
    from pydantic import BaseSettings
    AliasChoices = None  # type: ignore

# 自定义 env source：对 Agent 触发白名单这种"CSV 风格"的列表字段
# 关闭默认的 JSON 解码，将原始字符串透传给 _split_csv field_validator
try:
    from pydantic_settings.sources import EnvSettingsSource as _EnvSettingsSource
except ImportError:  # 极旧版本兜底
    _EnvSettingsSource = None  # type: ignore

# 这些字段在 ENV 中以"逗号分隔字符串"形式出现，应交由 _split_csv 处理
# 而非由 pydantic-settings 默认尝试 JSON 解析
_CSV_FIELDS_FOR_LIST = {
    "agent_trigger_query_types",
    "agent_trigger_evidence_needs",
    "memory_shared_mode_allowed_kinds",
}

logger = logging.getLogger(__name__)


if _EnvSettingsSource is not None:

    class _CsvAwareEnvSettingsSource(_EnvSettingsSource):
        """对 _CSV_FIELDS_FOR_LIST 中的字段跳过 JSON 解码，直接返回原始字符串，
        由 AppSettings 上 ``mode="before"`` 的 _split_csv 校验器完成 CSV 拆分。"""

        def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
            if field_name in _CSV_FIELDS_FOR_LIST and isinstance(value, str):
                return value
            return super().decode_complex_value(field_name, field, value)
else:  # pragma: no cover - 仅极旧 pydantic_settings 才会落到此分支
    _CsvAwareEnvSettingsSource = None  # type: ignore


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
    error_log_path: str = Field(default="", env="CHATPDF_ERROR_LOG_PATH")
    # 超时/断路器
    chat_timeout: float = Field(default=120.0, env="CHATPDF_CHAT_TIMEOUT")
    search_timeout: float = Field(default=30.0, env="CHATPDF_SEARCH_TIMEOUT")

    # 备用模型/提供商（用于失败兜底）
    chat_fallback_provider: str | None = Field(default=None, env="CHATPDF_CHAT_FALLBACK_PROVIDER")
    chat_fallback_model: str | None = Field(default=None, env="CHATPDF_CHAT_FALLBACK_MODEL")
    search_fallback_provider: str | None = Field(default=None, env="CHATPDF_SEARCH_FALLBACK_PROVIDER")
    search_fallback_model: str | None = Field(default=None, env="CHATPDF_SEARCH_FALLBACK_MODEL")

    # ==================== Agent 检索配置 ====================
    # Agent 检索最大轮数
    agent_max_rounds: int = Field(
        default=3,  # P1.4 优化：5→3，实测 2-3 轮足够获取概览类上下文，减少 planner 串行延迟
        validation_alias=AliasChoices("agent_max_rounds", "CHATPDF_AGENT_MAX_ROUNDS"),
        description="Agent 检索最大轮数，范围 1-10（默认 3，复杂问题可调高）"
    )
    # Agent planner 模型温度
    agent_planner_temperature: float = Field(
        default=0.3,
        validation_alias=AliasChoices("agent_planner_temperature", "CHATPDF_AGENT_PLANNER_TEMPERATURE"),
        description="Agent planner LLM 温度参数"
    )
    agent_planner_retries: int = Field(
        default=1,
        validation_alias=AliasChoices("agent_planner_retries", "CHATPDF_AGENT_PLANNER_RETRIES"),
        description="Agent planner JSON 解析失败后的重试次数"
    )
    agent_context_max_tokens: int = Field(
        default=12000,
        validation_alias=AliasChoices("agent_context_max_tokens", "CHATPDF_AGENT_CONTEXT_MAX_TOKENS"),
        description="Agent 最终检索上下文最大 token 数"
    )
    agent_max_iterations: int = Field(
        default=3,
        validation_alias=AliasChoices("agent_max_iterations", "CHATPDF_AGENT_MAX_ITERATIONS"),
        description="Agent 检索最大迭代次数，范围 1-10"
    )
    agent_max_tool_calls: int = Field(
        default=12,
        validation_alias=AliasChoices("agent_max_tool_calls", "CHATPDF_AGENT_MAX_TOOL_CALLS"),
        description="Agent 单次检索最大工具调用数"
    )
    agent_context_compress_threshold: int = Field(
        default=16000,
        validation_alias=AliasChoices("agent_context_compress_threshold", "CHATPDF_AGENT_CONTEXT_COMPRESS_THRESHOLD"),
        description="Agent planner 已取材上下文压缩阈值（字符数）"
    )
    agent_tool_max_concurrency: int = Field(
        default=5,
        validation_alias=AliasChoices("agent_tool_max_concurrency", "CHATPDF_AGENT_TOOL_MAX_CONCURRENCY"),
        description="Agent 同轮工具最大并发数，范围 1-5"
    )
    # Agent planner 单次 LLM 调用超时（秒），防止单次模型推理卡死整个 Agent
    # 参考 agentic-rag-for-dummies 的硬上限思路；不设置 ENV 默认 30 秒
    agent_planner_timeout: float = Field(
        default=30.0,
        validation_alias=AliasChoices("agent_planner_timeout", "CHATPDF_AGENT_PLANNER_TIMEOUT"),
        description="Agent planner 单次 LLM 调用的最大等待秒数，最小 15s"
    )
    agent_total_timeout: float = Field(
        default=75.0,
        validation_alias=AliasChoices("agent_total_timeout", "CHATPDF_AGENT_TOTAL_TIMEOUT"),
        description="Agent 单次检索整体最大等待秒数，超时后使用已取材内容降级"
    )
    # Agent 单次工具调用超时（秒），参考 ragflow `COMPONENT_EXEC_TIMEOUT=12`
    # 单工具卡住时仅丢弃该工具结果，不拖垮整轮 Agent；候选池 trace 仍能积累。
    agent_tool_timeout: float = Field(
        default=12.0,
        validation_alias=AliasChoices("agent_tool_timeout", "CHATPDF_AGENT_TOOL_TIMEOUT"),
        description="Agent 单次工具调用的最大等待秒数，超时后该工具返回空结果但不影响其他工具"
    )
    agent_external_rerank_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("agent_external_rerank_enabled", "CHATPDF_AGENT_EXT_RERANK"),
        description="是否允许 Agent 在最终上下文构造阶段调用外部/本地 rerank_service 做终选重排"
    )
    # paper-qa 风格证据级 LLM 评分：按条打 summary + relevance_score，再分层消费
    agent_evidence_scoring_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "agent_evidence_scoring_enabled",
            "CHATPDF_AGENT_EVIDENCE_SCORING",
        ),
        description="Agent 是否对意群/chunk 级证据做 LLM 相关性评分（exact table 直通）",
    )
    agent_evidence_scoring_timeout: float = Field(
        default=8.0,
        validation_alias=AliasChoices(
            "agent_evidence_scoring_timeout",
            "CHATPDF_AGENT_EVIDENCE_SCORING_TIMEOUT",
        ),
        description="证据评分整批超时秒数，超时后回退启发式充足度判断",
    )
    agent_evidence_scoring_min_candidates: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "agent_evidence_scoring_min_candidates",
            "CHATPDF_AGENT_EVIDENCE_SCORING_MIN",
        ),
        description="非 exact 候选少于此数时跳过 LLM 评分",
    )
    agent_evidence_k: int = Field(
        default=10,
        validation_alias=AliasChoices("agent_evidence_k", "CHATPDF_AGENT_EVIDENCE_K"),
        description="进入 LLM 评分的最大证据候选数",
    )
    agent_answer_max_sources: int = Field(
        default=8,
        validation_alias=AliasChoices(
            "agent_answer_max_sources",
            "CHATPDF_AGENT_ANSWER_MAX_SOURCES",
        ),
        description="评分后进入最终上下文的最大证据条数",
    )
    # Agent 模式模糊问句的 cheap-model 澄清判定（规则层之后）
    agent_llm_clarification_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "agent_llm_clarification_enabled",
            "CHATPDF_AGENT_LLM_CLARIFICATION",
        ),
        description="Agent 路径是否用 cheap model 二次判定问句是否足够清晰",
    )
    # is_ambiguous 为真时如何影响这一轮：
    #   off       —— 完全忽略，不产生任何用户可见影响
    #   hint      —— 检索照常执行，仅附带 clarification_hint（默认，fail-open）
    #   interrupt —— 旧行为：直接返回澄清问句、跳过检索（向后兼容）
    intent_clarification_mode: str = Field(
        default="hint",
        validation_alias=AliasChoices(
            "intent_clarification_mode",
            "CHATPDF_INTENT_CLARIFICATION_MODE",
        ),
        description="模糊意图处理模式：off / hint / interrupt",
    )
    # 意图判定 trace 落盘（只观测不干预）。关闭时不产生任何文件 IO。
    intent_trace_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "intent_trace_enabled",
            "CHATPDF_INTENT_TRACE_ENABLED",
        ),
        description="是否把每轮意图判定的判据落成 JSONL trace（默认关闭，避免保留会话数据）",
    )
    intent_trace_include_question_preview: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "intent_trace_include_question_preview",
            "CHATPDF_INTENT_TRACE_INCLUDE_QUESTION_PREVIEW",
        ),
        description="启用意图 trace 时是否额外保留问题前缀；默认仅保存不可逆 hash",
    )
    intent_trace_retention_days: int = Field(
        default=14,
        validation_alias=AliasChoices(
            "intent_trace_retention_days",
            "CHATPDF_INTENT_TRACE_RETENTION_DAYS",
        ),
        description="意图 trace 保留天数，0 表示不按日期清理",
    )
    intent_trace_max_total_bytes: int = Field(
        default=100 * 1024 * 1024,
        validation_alias=AliasChoices(
            "intent_trace_max_total_bytes",
            "CHATPDF_INTENT_TRACE_MAX_TOTAL_BYTES",
        ),
        description="意图 trace 目录总容量上限，超出时优先删除最旧文件",
    )

    # ==================== Agent 触发白名单与增强开关 ====================
    # 仅概览/分析类问题默认直接进入 Agent。具体问答由 evidence_need 决定，
    # 数值抽取保留给表格视觉核验链路；ENV 可传逗号分隔字符串覆盖。
    agent_trigger_query_types: list[str] = Field(
        default=["overview", "analytical"],
        validation_alias=AliasChoices(
            "agent_trigger_query_types",
            "AGENT_TRIGGER_QUERY_TYPES",
        ),
        description="允许进入 Agent 路径的 query_type 集合（CSV 字符串自动拆分）"
    )
    # 允许进入 Agent 路径的 evidence_need 集合；ENV 可传逗号分隔字符串
    agent_trigger_evidence_needs: list[str] = Field(
        default=["section_explanation", "comparison_multi_aspect", "reference_meta", "analysis_explanation"],
        validation_alias=AliasChoices(
            "agent_trigger_evidence_needs",
            "AGENT_TRIGGER_EVIDENCE_NEEDS",
        ),
        description="允许进入 Agent 路径的 evidence_need 集合"
    )
    # 是否在每轮命中后自动回填父级语义组 digest
    enable_parent_backfill: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_parent_backfill",
            "AGENT_ENABLE_PARENT_BACKFILL",
        ),
        description="是否在每轮命中后自动回填父级语义组 digest"
    )
    # Planner_LLM 是否优先使用原生 function calling
    use_native_tools: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "use_native_tools",
            "AGENT_USE_NATIVE_TOOLS",
        ),
        description="Planner_LLM 是否优先使用原生 function calling，不支持时回退 JSON 文本解析"
    )

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
    # Deprecated compatibility fields. Online document parsing is MinerU-only;
    # these values are read only when migrating an older installation.
    mistral_ocr_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("mistral_ocr_api_key", "CHATPDF_MISTRAL_OCR_API_KEY"),
        description="旧版 Mistral OCR API Key（仅迁移兼容）"
    )
    mistral_ocr_base_url: str = Field(
        default="https://api.mistral.ai",
        validation_alias=AliasChoices("mistral_ocr_base_url", "CHATPDF_MISTRAL_OCR_BASE_URL"),
        description="旧版 Mistral OCR Base URL（仅迁移兼容）"
    )

    # ==================== Figure Extraction 配置 ====================
    # Figure Extraction 超时配置（秒）
    figure_extraction_timeout_sec: int = Field(
        default=15,
        validation_alias=AliasChoices("figure_extraction_timeout_sec", "CHATPDF_FIGURE_EXTRACTION_TIMEOUT"),
        description="Figure Extraction 超时时间（秒）"
    )

    # ==================== GraphRAG 配置 ====================
    # 是否启用 GraphRAG 知识图谱增强检索
    enable_graphrag: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_graphrag", "CHATPDF_ENABLE_GRAPHRAG"),
        description="是否启用 GraphRAG 知识图谱增强检索"
    )
    # GraphRAG 工作目录（相对于 data/）
    graphrag_working_dir: str = Field(
        default="data/graphrag",
        validation_alias=AliasChoices("graphrag_working_dir", "CHATPDF_GRAPHRAG_WORKING_DIR"),
        description="GraphRAG 持久化工作目录"
    )
    # GraphRAG 实体提取 gleaning 轮数（0=不做追加提取，1=追加一轮）
    graphrag_max_gleaning: int = Field(
        default=1,
        validation_alias=AliasChoices("graphrag_max_gleaning", "CHATPDF_GRAPHRAG_MAX_GLEANING"),
        description="实体提取 gleaning 轮数，范围 0-3"
    )
    # GraphRAG 分块 token 大小
    graphrag_chunk_token_size: int = Field(
        default=2000,
        validation_alias=AliasChoices("graphrag_chunk_token_size", "CHATPDF_GRAPHRAG_CHUNK_TOKEN_SIZE"),
        description="GraphRAG 分块 token 大小"
    )
    # GraphRAG LLM 最大并发数
    graphrag_max_async: int = Field(
        default=8,
        validation_alias=AliasChoices("graphrag_max_async", "CHATPDF_GRAPHRAG_MAX_ASYNC"),
        description="GraphRAG LLM 最大并发请求数"
    )
    # GraphRAG 查询模式: auto / local / global / hybrid
    graphrag_query_mode: str = Field(
        default="auto",
        validation_alias=AliasChoices("graphrag_query_mode", "CHATPDF_GRAPHRAG_QUERY_MODE"),
        description="GraphRAG 查询模式: auto(按query_type自动选择) / local / global / hybrid"
    )
    # GraphRAG 上下文最大 token 数（防止拼接后超出 LLM 上下文窗口）
    graphrag_context_max_tokens: int = Field(
        default=2000,
        validation_alias=AliasChoices("graphrag_context_max_tokens", "CHATPDF_GRAPHRAG_CONTEXT_MAX_TOKENS"),
        description="GraphRAG 上下文最大 token 数，超限时截断"
    )

    # ==================== BM25 分词配置 ====================
    # 是否使用 jieba 分词增强 BM25（中文检索质量更高）
    bm25_use_jieba: bool = Field(
        default=True,
        validation_alias=AliasChoices("bm25_use_jieba", "CHATPDF_BM25_USE_JIEBA"),
        description="是否使用 jieba 分词增强 BM25 中文检索"
    )

    # 是否启用 BM25 同义词扩展（查询时自动扩展同义词，提升召回率）
    bm25_expand_synonyms: bool = Field(
        default=True,
        validation_alias=AliasChoices("bm25_expand_synonyms", "CHATPDF_BM25_EXPAND_SYNONYMS"),
        description="是否启用 BM25 同义词扩展"
    )
    # 自定义同义词词典路径（JSON 格式，可选）
    bm25_synonym_dict_path: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("bm25_synonym_dict_path", "CHATPDF_BM25_SYNONYM_DICT_PATH"),
        description="自定义同义词词典文件路径（JSON 格式）"
    )

    # ==================== 上下文扩展配置 ====================
    # 命中 chunk 邻居扩展数（前后各扩展 N 个 chunk）
    num_expand_context_chunk: int = Field(
        default=1,
        validation_alias=AliasChoices("num_expand_context_chunk", "CHATPDF_NUM_EXPAND_CONTEXT_CHUNK"),
        description="命中 chunk 前后各扩展的邻居 chunk 数，0=不扩展"
    )

    # ==================== 双模型策略配置 ====================
    # 辅助模型：用于查询改写、问题分解、追问生成等非核心 LLM 任务
    # 留空则使用用户选择的主模型（与之前行为一致）
    cheap_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("cheap_model", "CHATPDF_CHEAP_MODEL"),
        description="辅助模型名称（如 gpt-4o-mini），用于非核心 LLM 任务，留空使用主模型"
    )
    cheap_model_provider: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("cheap_model_provider", "CHATPDF_CHEAP_MODEL_PROVIDER"),
        description="辅助模型提供商（如 openai），留空使用主模型提供商"
    )

    # ==================== 查询改写配置 ====================
    # 是否启用 LLM 查询改写（多轮对话指代消解）
    enable_llm_query_rewrite: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable_llm_query_rewrite", "CHATPDF_ENABLE_LLM_QUERY_REWRITE"),
        description="是否启用 LLM 查询改写"
    )
    # 查询长度阈值：超过此长度的查询不做 LLM 改写（信息已足够）
    query_rewrite_trigger_length: int = Field(
        default=150,
        validation_alias=AliasChoices("query_rewrite_trigger_length", "CHATPDF_QUERY_REWRITE_TRIGGER_LENGTH"),
        description="触发 LLM 改写的最大查询长度"
    )

    # ==================== 答案自审配置 ====================
    # 是否启用答案自审（用 cheap model 检测幻觉，默认关闭以减少延迟）
    enable_answer_critic: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_answer_critic", "CHATPDF_ENABLE_ANSWER_CRITIC"),
        description="是否启用答案自审（检测幻觉，增加延迟）"
    )
    enable_answer_critic_repair: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_answer_critic_repair",
            "CHATPDF_ENABLE_ANSWER_CRITIC_REPAIR",
        ),
        description="答案自审发现高风险时，非流式模式是否使用同一证据做一次受控修复",
    )
    enable_reading_outline_claim_verifier: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "enable_reading_outline_claim_verifier",
            "CHATPDF_ENABLE_READING_OUTLINE_CLAIM_VERIFIER",
        ),
        description="是否对阅读总结中的关键结果、强比较和限制条件做选择性语义核验",
    )
    enable_reading_outline_visual_preflight: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_reading_outline_visual_preflight",
            "CHATPDF_ENABLE_READING_OUTLINE_VISUAL_PREFLIGHT",
        ),
        description="生成阅读总结前是否限额补充方法、实验和消融章节中的高风险图表",
    )
    reading_outline_visual_preflight_limit: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "reading_outline_visual_preflight_limit",
            "CHATPDF_READING_OUTLINE_VISUAL_PREFLIGHT_LIMIT",
        ),
        description="阅读总结视觉预检最多分析的高风险图表数（服务内硬限制 1-4）",
    )
    enable_paper_metadata_hydration: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "enable_paper_metadata_hydration",
            "CHATPDF_ENABLE_PAPER_METADATA_HYDRATION",
        ),
        description="是否在文档读取后异步补全外部论文元数据；默认关闭以保护隐私",
    )
    paper_metadata_hydration_timeout_seconds: float = Field(
        default=8.0,
        validation_alias=AliasChoices(
            "paper_metadata_hydration_timeout_seconds",
            "CHATPDF_PAPER_METADATA_HYDRATION_TIMEOUT_SECONDS",
        ),
        description="单个外部元数据请求超时秒数",
    )
    paper_metadata_unpaywall_email: str = Field(
        default="",
        validation_alias=AliasChoices(
            "paper_metadata_unpaywall_email",
            "CHATPDF_PAPER_METADATA_UNPAYWALL_EMAIL",
        ),
        description="可选 Unpaywall 联系邮箱；为空时跳过该 provider",
    )
    paper_metadata_semantic_scholar_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "paper_metadata_semantic_scholar_api_key",
            "CHATPDF_PAPER_METADATA_SEMANTIC_SCHOLAR_API_KEY",
        ),
        description="可选 Semantic Scholar API Key",
    )

    # ==================== P3.6 引用增强（two-pass citation injection）====================
    # 启用后，answer 引用覆盖率 < 阈值时会触发二次 LLM 调用补全 [N] 引文
    enable_citation_enhancer: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable_citation_enhancer", "CHATPDF_ENABLE_CITATION_ENHANCER"),
        description="是否启用 citation_plus 二次引用注入（提升 faithfulness，增加 3-5s 延迟）"
    )
    citation_enhancer_coverage_threshold: float = Field(
        default=0.7,
        validation_alias=AliasChoices(
            "citation_enhancer_coverage_threshold",
            "CHATPDF_CITATION_ENHANCER_COVERAGE_THRESHOLD",
        ),
        description="原答案引用覆盖率低于此阈值才触发二次注入，范围 0.0-1.0；参考 ragflow 默认对所有答案做二次对齐，Chatpdf 取 0.7 平衡延迟与 selector gap"
    )
    # ==================== P3.4 / P3.5 ablation flags（默认 ON，仅评估时关闭）====================
    enable_p34_layered_context: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_p34_layered_context",
            "CHATPDF_ENABLE_P34_LAYERED_CONTEXT",
        ),
        description="P3.4 三层上下文格式化总开关（关闭则不重组 context_string，用于 ablation）"
    )
    enable_p35_citation_prompt: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_p35_citation_prompt",
            "CHATPDF_ENABLE_P35_CITATION_PROMPT",
        ),
        description="P3.5 详细引用规则手册总开关（关闭则不注入 _build_faithfulness_guard_prompt，用于 ablation）"
    )

    # ==================== P3.7 树式递归查询分解 + sufficiency_check ====================
    # 启用后，对 analytical/overview/comparison 类问题，首轮检索 < 5 chunks 时
    # 触发 sufficiency_check + multi-query gen + 并行递归子查询
    enable_tree_decomposition: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_tree_decomposition", "CHATPDF_ENABLE_TREE_DECOMPOSITION"),
        description="是否启用树式递归查询分解（提升复杂查询召回，仅在首轮 chunks < 5 时触发）"
    )
    tree_decomposition_max_depth: int = Field(
        default=2,
        validation_alias=AliasChoices(
            "tree_decomposition_max_depth",
            "CHATPDF_TREE_DECOMPOSITION_MAX_DEPTH",
        ),
        description="树式递归最大深度，建议 1-3"
    )
    tree_decomposition_k_per_level: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "tree_decomposition_k_per_level",
            "CHATPDF_TREE_DECOMPOSITION_K_PER_LEVEL",
        ),
        description="每层生成的子查询数，建议 2-5"
    )

    # ==================== numeric_table 专项检索增强 ====================
    # 是否启用 numeric_table 专项检索增强（针对表格数值比较类查询，如
    # "第二好的方法"、"Table N | Method | Metric=value"）
    # 关闭后：不做 numeric_table hard gate / evidence slot / structured
    # bundle 升级等专项处理，走通用主链路。用于 A/B 验证专项逻辑对通用
    # 场景（纯文本、overview、非表格 extraction）是否存在退化影响。
    enable_numeric_table_specialization: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_numeric_table_specialization",
            "CHATPDF_ENABLE_NUMERIC_TABLE",
        ),
        description="是否启用 numeric_table 专项检索增强，关闭则走通用主链路"
    )

    # ==================== 流式输出缓冲配置 ====================
    # 流式输出缓冲字符数阈值，累积超过此值后发送，0 表示禁用缓冲（直通模式）
    stream_buffer_size: int = Field(
        default=5,
        validation_alias=AliasChoices("stream_buffer_size", "CHATPDF_STREAM_BUFFER_SIZE"),
        description="流式输出缓冲字符数阈值，累积超过此值后发送，0 表示禁用缓冲"
    )

    # ==================== 记忆系统配置 ====================
    # 记忆功能启用开关
    memory_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("memory_enabled", "CHATPDF_MEMORY_ENABLED"),
        description="记忆功能启用开关"
    )
    # 关键词频率阈值，超过此值的关键词将被识别为用户关注领域
    memory_keyword_threshold: int = Field(
        default=3,
        validation_alias=AliasChoices("memory_keyword_threshold", "CHATPDF_MEMORY_KEYWORD_THRESHOLD"),
        description="关键词频率阈值，范围 1-100"
    )
    # QA 摘要保留上限
    memory_max_summaries: int = Field(
        default=50,
        validation_alias=AliasChoices("memory_max_summaries", "CHATPDF_MEMORY_MAX_SUMMARIES"),
        description="QA 摘要保留上限，范围 1-1000"
    )
    # 记忆检索返回条数
    memory_retrieval_top_k: int = Field(
        default=3,
        validation_alias=AliasChoices("memory_retrieval_top_k", "CHATPDF_MEMORY_RETRIEVAL_TOP_K"),
        description="记忆检索返回条数，范围 1-20"
    )
    # 是否使用 SQLite 存储（可选增强，默认 False 保持向后兼容）
    memory_use_sqlite: bool = Field(
        default=False,
        validation_alias=AliasChoices("memory_use_sqlite", "CHATPDF_MEMORY_USE_SQLITE"),
        description="是否使用 SQLite 存储记忆（提供更好的查询性能）"
    )
    # Pre-compaction 记忆刷新配置
    memory_flush_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("memory_flush_enabled", "CHATPDF_MEMORY_FLUSH_ENABLED"),
        description="是否启用 pre-compaction 记忆刷新"
    )
    # Pre-compaction 刷新阈值（token 数）
    memory_flush_threshold_tokens: int = Field(
        default=4000,
        validation_alias=AliasChoices("memory_flush_threshold_tokens", "CHATPDF_MEMORY_FLUSH_THRESHOLD_TOKENS"),
        description="触发记忆刷新的 token 阈值，范围 1000-20000"
    )
    # 工作记忆窗口大小（保留最近 N 轮对话）
    memory_working_window_size: int = Field(
        default=10,
        validation_alias=AliasChoices("memory_working_window_size", "CHATPDF_MEMORY_WORKING_WINDOW_SIZE"),
        description="工作记忆窗口大小，范围 1-50"
    )
    # 晋升命中次数阈值（短期记忆晋升为长期记忆所需的命中次数）
    memory_promotion_threshold: int = Field(
        default=5,
        validation_alias=AliasChoices("memory_promotion_threshold", "CHATPDF_MEMORY_PROMOTION_THRESHOLD"),
        description="晋升命中次数阈值，范围 1-100"
    )
    # 降级天数阈值（长期记忆超过此天数未命中则降级）
    memory_demotion_days: int = Field(
        default=90,
        validation_alias=AliasChoices("memory_demotion_days", "CHATPDF_MEMORY_DEMOTION_DAYS"),
        description="降级天数阈值，范围 1-365"
    )
    # 压缩触发条目数（同一文档记忆超过此数量触发压缩）
    memory_compression_threshold: int = Field(
        default=20,
        validation_alias=AliasChoices("memory_compression_threshold", "CHATPDF_MEMORY_COMPRESSION_THRESHOLD"),
        description="压缩触发条目数，范围 5-200"
    )
    # 压缩后最多保留的条目数
    memory_compression_max_items: int = Field(
        default=5,
        validation_alias=AliasChoices("memory_compression_max_items", "CHATPDF_MEMORY_COMPRESSION_MAX_ITEMS"),
        description="压缩后最多保留几条记忆，范围 1-20"
    )
    # 单条记忆超过此字符数视为超长，压缩失败时优先剔除
    memory_compression_oversized_chars: int = Field(
        default=2000,
        validation_alias=AliasChoices("memory_compression_oversized_chars", "CHATPDF_MEMORY_COMPRESSION_OVERSIZED_CHARS"),
        description="超长记忆判定字符数，范围 200-20000"
    )
    # 单文档记忆的存储配额（字符数，0 表示不限制）
    memory_quota_chars_per_doc: int = Field(
        default=200_000,
        validation_alias=AliasChoices("memory_quota_chars_per_doc", "CHATPDF_MEMORY_QUOTA_CHARS_PER_DOC"),
        description="单文档记忆内容总字符上限，0 表示不限制；条数阈值对长短记忆一视同仁，配额才给得出存储上界"
    )
    # 共享/演示场景下允许注入的记忆层白名单（默认排除个人画像与跨文档对话摘要）
    memory_shared_mode_allowed_kinds: list[str] = Field(
        default_factory=lambda: ["working", "doc_fact", "consolidated", "graph"],
        validation_alias=AliasChoices("memory_shared_mode_allowed_kinds", "CHATPDF_MEMORY_SHARED_MODE_ALLOWED_KINDS"),
        description="共享模式下可注入的记忆层，逗号分隔；profile/episodic 默认被排除"
    )
    # 证据评分缓存：同一 (解析身份, 问题, 证据) 的 summary+score 复用
    evidence_score_cache_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("evidence_score_cache_enabled", "CHATPDF_EVIDENCE_SCORE_CACHE_ENABLED"),
        description="重新生成/重试/多轮迭代时复用证据摘要，跳过重复的 LLM 打分"
    )
    evidence_score_cache_size: int = Field(
        default=2000,
        validation_alias=AliasChoices("evidence_score_cache_size", "CHATPDF_EVIDENCE_SCORE_CACHE_SIZE"),
        description="证据评分缓存条目上限，范围 100-50000"
    )
    # 全部证据（文档 + 联网 + 记忆 + 术语）的统一 token 预算
    memory_evidence_total_budget: int = Field(
        default=13000,
        validation_alias=AliasChoices("memory_evidence_total_budget", "CHATPDF_MEMORY_EVIDENCE_TOTAL_BUDGET"),
        description="证据上下文总预算；记忆在其中取剩余额度，范围 1000-100000"
    )
    # 记忆的保底注入额度：文档再长也要留一点，否则长文档会话里记忆等于被静默关掉
    memory_injection_floor_tokens: int = Field(
        default=200,
        validation_alias=AliasChoices("memory_injection_floor_tokens", "CHATPDF_MEMORY_INJECTION_FLOOR_TOKENS"),
        description="记忆注入的保底 token 数，范围 0-2000"
    )
    # 单轮记忆写入允许发起的后台 LLM 调用总数上限
    memory_llm_calls_per_turn: int = Field(
        default=3,
        validation_alias=AliasChoices("memory_llm_calls_per_turn", "CHATPDF_MEMORY_LLM_CALLS_PER_TURN"),
        description="一轮对话里记忆链路最多发起几次后台 LLM 调用（提炼/裁决/压缩/会话摘要/图谱按此优先级消费），范围 0-10；0 表示禁用全部记忆 LLM 调用"
    )
    # 滚动会话摘要：补上 working 层与长期记忆之间的中期叙事连续性
    memory_session_summary_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("memory_session_summary_enabled", "CHATPDF_MEMORY_SESSION_SUMMARY_ENABLED"),
        description="维护每文档一条滚动会话摘要，长会话里保住指代链"
    )
    # 每积累多少条消息更新一次滚动摘要
    memory_session_summary_interval: int = Field(
        default=6,
        validation_alias=AliasChoices("memory_session_summary_interval", "CHATPDF_MEMORY_SESSION_SUMMARY_INTERVAL"),
        description="滚动摘要更新的消息条数间隔，范围 2-50"
    )
    # 用 LLM 从记忆事实里抽取实体关系三元组构建论文图谱
    memory_graph_llm_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("memory_graph_llm_enabled", "CHATPDF_MEMORY_GRAPH_LLM_ENABLED"),
        description="关闭后图谱退回只抓 Figure/Table 引用的正则版本"
    )
    # 事实数变化超过此增量才重建图谱，用于摊薄 LLM 成本
    memory_graph_rebuild_delta: int = Field(
        default=5,
        validation_alias=AliasChoices("memory_graph_rebuild_delta", "CHATPDF_MEMORY_GRAPH_REBUILD_DELTA"),
        description="图谱重建的事实增量阈值，范围 1-100"
    )
    # 归档回捞（page fault）：活跃记忆没填满配额时从归档里补检
    memory_archive_recall_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("memory_archive_recall_enabled", "CHATPDF_MEMORY_ARCHIVE_RECALL_ENABLED"),
        description="压缩归档的原始记忆是否可在召回不足时被回捞"
    )
    # 归档回捞的最小 token 重叠率
    memory_archive_recall_min_overlap: float = Field(
        default=0.3,
        validation_alias=AliasChoices("memory_archive_recall_min_overlap", "CHATPDF_MEMORY_ARCHIVE_RECALL_MIN_OVERLAP"),
        description="归档回捞的查询 token 最小重叠率，范围 0-1"
    )
    # 活跃记忆池容量（类 OS RAM，LRU 策略管理）
    memory_active_pool_size: int = Field(
        default=100,
        validation_alias=AliasChoices("memory_active_pool_size", "CHATPDF_MEMORY_ACTIVE_POOL_SIZE"),
        description="活跃记忆池容量，范围 10-1000"
    )
    # 注入 token 预算（记忆注入到 system prompt 的最大 token 数）
    memory_injection_token_budget: int = Field(
        default=800,
        validation_alias=AliasChoices("memory_injection_token_budget", "CHATPDF_MEMORY_INJECTION_TOKEN_BUDGET"),
        description="注入 token 预算，范围 100-5000"
    )
    # 分层注入配额：留空则使用 ContextInjector 内置默认值
    memory_injection_kind_budgets: dict[str, int] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("memory_injection_kind_budgets", "CHATPDF_MEMORY_INJECTION_KIND_BUDGETS"),
        description='各记忆层 token 配额，JSON 对象如 {"doc_fact": 260}；留空用内置默认'
    )
    # 记忆检索软超时（秒），0 表示不限时
    memory_retrieval_timeout: float = Field(
        default=2.0,
        validation_alias=AliasChoices("memory_retrieval_timeout", "CHATPDF_MEMORY_RETRIEVAL_TIMEOUT"),
        description="记忆检索软超时秒数，范围 0-30，0 表示不限时"
    )
    # 记忆后台写线程并发上限
    memory_background_max_pending: int = Field(
        default=6,
        validation_alias=AliasChoices("memory_background_max_pending", "CHATPDF_MEMORY_BACKGROUND_MAX_PENDING"),
        description="记忆后台写线程并发上限，范围 1-32"
    )
    # 写入去重的向量相似度阈值，>=1.0 表示关闭相似度去重
    memory_dedup_similarity_threshold: float = Field(
        default=0.85,
        validation_alias=AliasChoices("memory_dedup_similarity_threshold", "CHATPDF_MEMORY_DEDUP_SIMILARITY_THRESHOLD"),
        description="新记忆与既有记忆的相似度超过此值则拒写，范围 0-1，1.0 关闭"
    )
    # 是否启用写入裁决（新事实与既有记忆比对后决定 ADD/UPDATE/DELETE/NONE）
    memory_arbitration_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("memory_arbitration_enabled", "CHATPDF_MEMORY_ARBITRATION_ENABLED"),
        description="启用后每轮记忆写入会多一次 LLM 调用，换取去重与冲突消解"
    )
    # 用户显式记忆指令（"记住…"）是否直接晋升为长期高价值记忆
    memory_explicit_promotion_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("memory_explicit_promotion_enabled", "CHATPDF_MEMORY_EXPLICIT_PROMOTION_ENABLED"),
        description="用户显式要求记住的内容是否跳过命中计数直接晋升为长期记忆"
    )

    @field_validator(
        "agent_trigger_query_types",
        "agent_trigger_evidence_needs",
        "memory_shared_mode_allowed_kinds",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, v):
        """支持环境变量以逗号分隔字符串形式覆盖列表配置；
        空白与重复逗号被忽略；非字符串类型直接透传。"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("memory_keyword_threshold")
    @classmethod
    def validate_memory_keyword_threshold(cls, v: int) -> int:
        """校验关键词频率阈值，范围 1-100，超出范围使用默认值"""
        if not (1 <= v <= 100):
            logger.warning(
                f"memory_keyword_threshold 值 {v} 超出合理范围 (1-100)，使用默认值 3"
            )
            return 3
        return v

    @field_validator("memory_max_summaries")
    @classmethod
    def validate_memory_max_summaries(cls, v: int) -> int:
        """校验 QA 摘要保留上限，范围 1-1000，超出范围使用默认值"""
        if not (1 <= v <= 1000):
            logger.warning(
                f"memory_max_summaries 值 {v} 超出合理范围 (1-1000)，使用默认值 50"
            )
            return 50
        return v

    @field_validator("memory_retrieval_top_k")
    @classmethod
    def validate_memory_retrieval_top_k(cls, v: int) -> int:
        """校验记忆检索返回条数，范围 1-20，超出范围使用默认值"""
        if not (1 <= v <= 20):
            logger.warning(
                f"memory_retrieval_top_k 值 {v} 超出合理范围 (1-20)，使用默认值 3"
            )
            return 3
        return v

    @field_validator("memory_flush_threshold_tokens")
    @classmethod
    def validate_memory_flush_threshold_tokens(cls, v: int) -> int:
        """校验记忆刷新阈值，范围 1000-20000，超出范围使用默认值"""
        if not (1000 <= v <= 20000):
            logger.warning(
                f"memory_flush_threshold_tokens 值 {v} 超出合理范围 (1000-20000)，使用默认值 4000"
            )
            return 4000
        return v

    @field_validator("memory_working_window_size")
    @classmethod
    def validate_memory_working_window_size(cls, v: int) -> int:
        """校验工作记忆窗口大小，范围 1-50，超出范围使用默认值"""
        if not (1 <= v <= 50):
            logger.warning(
                f"memory_working_window_size 值 {v} 超出合理范围 (1-50)，使用默认值 10"
            )
            return 10
        return v

    @field_validator("memory_promotion_threshold")
    @classmethod
    def validate_memory_promotion_threshold(cls, v: int) -> int:
        """校验晋升命中次数阈值，范围 1-100，超出范围使用默认值"""
        if not (1 <= v <= 100):
            logger.warning(
                f"memory_promotion_threshold 值 {v} 超出合理范围 (1-100)，使用默认值 5"
            )
            return 5
        return v

    @field_validator("memory_demotion_days")
    @classmethod
    def validate_memory_demotion_days(cls, v: int) -> int:
        """校验降级天数阈值，范围 1-365，超出范围使用默认值"""
        if not (1 <= v <= 365):
            logger.warning(
                f"memory_demotion_days 值 {v} 超出合理范围 (1-365)，使用默认值 90"
            )
            return 90
        return v

    @field_validator("memory_compression_threshold")
    @classmethod
    def validate_memory_compression_threshold(cls, v: int) -> int:
        """校验压缩触发条目数，范围 5-200，超出范围使用默认值"""
        if not (5 <= v <= 200):
            logger.warning(
                f"memory_compression_threshold 值 {v} 超出合理范围 (5-200)，使用默认值 20"
            )
            return 20
        return v

    @field_validator("evidence_score_cache_size")
    @classmethod
    def validate_evidence_score_cache_size(cls, v: int) -> int:
        """校验证据评分缓存容量，范围 100-50000，超出范围使用默认值"""
        if not (100 <= v <= 50_000):
            logger.warning(
                f"evidence_score_cache_size 值 {v} 超出合理范围 (100-50000)，使用默认值 2000"
            )
            return 2000
        return v

    @field_validator("memory_evidence_total_budget")
    @classmethod
    def validate_memory_evidence_total_budget(cls, v: int) -> int:
        """校验证据总预算，范围 1000-100000，超出范围使用默认值"""
        if not (1000 <= v <= 100_000):
            logger.warning(
                f"memory_evidence_total_budget 值 {v} 超出合理范围 (1000-100000)，使用默认值 13000"
            )
            return 13000
        return v

    @field_validator("memory_injection_floor_tokens")
    @classmethod
    def validate_memory_injection_floor_tokens(cls, v: int) -> int:
        """校验记忆保底额度，范围 0-2000，超出范围使用默认值"""
        if not (0 <= v <= 2000):
            logger.warning(
                f"memory_injection_floor_tokens 值 {v} 超出合理范围 (0-2000)，使用默认值 200"
            )
            return 200
        return v

    @field_validator("memory_llm_calls_per_turn")
    @classmethod
    def validate_memory_llm_calls_per_turn(cls, v: int) -> int:
        """校验单轮记忆 LLM 调用上限，范围 0-10，超出范围使用默认值"""
        if not (0 <= v <= 10):
            logger.warning(
                f"memory_llm_calls_per_turn 值 {v} 超出合理范围 (0-10)，使用默认值 3"
            )
            return 3
        return v

    @field_validator("memory_session_summary_interval")
    @classmethod
    def validate_memory_session_summary_interval(cls, v: int) -> int:
        """校验滚动摘要更新间隔，范围 2-50，超出范围使用默认值"""
        if not (2 <= v <= 50):
            logger.warning(
                f"memory_session_summary_interval 值 {v} 超出合理范围 (2-50)，使用默认值 6"
            )
            return 6
        return v

    @field_validator("memory_graph_rebuild_delta")
    @classmethod
    def validate_memory_graph_rebuild_delta(cls, v: int) -> int:
        """校验图谱重建增量阈值，范围 1-100，超出范围使用默认值"""
        if not (1 <= v <= 100):
            logger.warning(
                f"memory_graph_rebuild_delta 值 {v} 超出合理范围 (1-100)，使用默认值 5"
            )
            return 5
        return v

    @field_validator("memory_archive_recall_min_overlap")
    @classmethod
    def validate_memory_archive_recall_min_overlap(cls, v: float) -> float:
        """校验归档回捞重叠率，范围 0-1，超出范围使用默认值"""
        if not (0.0 <= v <= 1.0):
            logger.warning(
                f"memory_archive_recall_min_overlap 值 {v} 超出合理范围 (0-1)，使用默认值 0.3"
            )
            return 0.3
        return v

    @field_validator("memory_compression_max_items")
    @classmethod
    def validate_memory_compression_max_items(cls, v: int) -> int:
        """校验压缩输出上限，范围 1-20，超出范围使用默认值"""
        if not (1 <= v <= 20):
            logger.warning(
                f"memory_compression_max_items 值 {v} 超出合理范围 (1-20)，使用默认值 5"
            )
            return 5
        return v

    @field_validator("memory_compression_oversized_chars")
    @classmethod
    def validate_memory_compression_oversized_chars(cls, v: int) -> int:
        """校验超长记忆判定阈值，范围 200-20000，超出范围使用默认值"""
        if not (200 <= v <= 20000):
            logger.warning(
                f"memory_compression_oversized_chars 值 {v} 超出合理范围 (200-20000)，使用默认值 2000"
            )
            return 2000
        return v

    @field_validator("memory_active_pool_size")
    @classmethod
    def validate_memory_active_pool_size(cls, v: int) -> int:
        """校验活跃记忆池容量，范围 10-1000，超出范围使用默认值"""
        if not (10 <= v <= 1000):
            logger.warning(
                f"memory_active_pool_size 值 {v} 超出合理范围 (10-1000)，使用默认值 100"
            )
            return 100
        return v

    @field_validator("memory_injection_token_budget")
    @classmethod
    def validate_memory_injection_token_budget(cls, v: int) -> int:
        """校验注入 token 预算，范围 100-5000，超出范围使用默认值"""
        if not (100 <= v <= 5000):
            logger.warning(
                f"memory_injection_token_budget 值 {v} 超出合理范围 (100-5000)，使用默认值 800"
            )
            return 800
        return v

    @field_validator("memory_retrieval_timeout")
    @classmethod
    def validate_memory_retrieval_timeout(cls, v: float) -> float:
        """校验记忆检索软超时，范围 0-30 秒，超出范围使用默认值"""
        if not (0.0 <= v <= 30.0):
            logger.warning(
                f"memory_retrieval_timeout 值 {v} 超出合理范围 (0-30)，使用默认值 2.0"
            )
            return 2.0
        return v

    @field_validator("memory_background_max_pending")
    @classmethod
    def validate_memory_background_max_pending(cls, v: int) -> int:
        """校验记忆后台写并发上限，范围 1-32，超出范围使用默认值"""
        if not (1 <= v <= 32):
            logger.warning(
                f"memory_background_max_pending 值 {v} 超出合理范围 (1-32)，使用默认值 6"
            )
            return 6
        return v

    @field_validator("memory_dedup_similarity_threshold")
    @classmethod
    def validate_memory_dedup_similarity_threshold(cls, v: float) -> float:
        """校验写入去重相似度阈值，范围 0-1，超出范围使用默认值"""
        if not (0.0 <= v <= 1.0):
            logger.warning(
                f"memory_dedup_similarity_threshold 值 {v} 超出合理范围 (0-1)，使用默认值 0.85"
            )
            return 0.85
        return v

    @field_validator("memory_injection_kind_budgets")
    @classmethod
    def validate_memory_injection_kind_budgets(cls, v: dict) -> dict:
        """丢弃非法的分层配额项，避免一个坏 key 让整份配置失效。"""
        if not isinstance(v, dict):
            logger.warning("memory_injection_kind_budgets 不是对象，已忽略")
            return {}
        cleaned: dict[str, int] = {}
        for kind, budget in v.items():
            try:
                budget_value = int(budget)
            except (TypeError, ValueError):
                logger.warning(f"memory_injection_kind_budgets['{kind}'] 非整数，已忽略")
                continue
            if budget_value < 0:
                logger.warning(f"memory_injection_kind_budgets['{kind}'] 为负数，已忽略")
                continue
            cleaned[str(kind)] = budget_value
        return cleaned

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

    @field_validator("intent_clarification_mode")
    @classmethod
    def validate_intent_clarification_mode(cls, v: str) -> str:
        """校验模糊意图处理模式；非法值降级为 hint 而不是让进程起不来。"""
        normalized = str(v or "").strip().lower()
        allowed = {"off", "hint", "interrupt"}
        if normalized not in allowed:
            logger.warning(
                f"intent_clarification_mode 值 {v!r} 非法（应为 {allowed} 之一），使用默认值 hint"
            )
            return "hint"
        return normalized

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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """注入 _CsvAwareEnvSettingsSource，使逗号分隔的 ENV 列表配置可被 _split_csv 处理。

        其他字段行为保持与默认 EnvSettingsSource 完全一致。"""
        if _CsvAwareEnvSettingsSource is None:
            return (init_settings, env_settings, dotenv_settings, file_secret_settings)
        custom_env = _CsvAwareEnvSettingsSource(
            settings_cls,
            case_sensitive=env_settings.case_sensitive,
            env_prefix=env_settings.env_prefix,
            env_nested_delimiter=env_settings.env_nested_delimiter,
            env_ignore_empty=env_settings.env_ignore_empty,
            env_parse_none_str=env_settings.env_parse_none_str,
            env_parse_enums=env_settings.env_parse_enums,
        )
        return (init_settings, custom_env, dotenv_settings, file_secret_settings)


settings = AppSettings()
