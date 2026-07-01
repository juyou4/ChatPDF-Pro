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
    error_log_path: str = Field(default="logs/errors.log", env="CHATPDF_ERROR_LOG_PATH")
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

    # ==================== Agent 触发白名单与增强开关 ====================
    # 允许进入 Agent 路径的 query_type 集合；ENV 可传逗号分隔字符串
    agent_trigger_query_types: list[str] = Field(
        default=["overview", "specific", "extraction", "analytical"],
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

    @field_validator(
        "agent_trigger_query_types",
        "agent_trigger_evidence_needs",
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
