"""
记忆管理核心服务

整合 MemoryStore、MemoryIndex、MemoryRetriever、KeywordExtractor，
提供统一的记忆管理业务接口。

核心功能：
- 记忆检索：检索相关记忆并返回格式化上下文
- 记忆写入：保存 QA 摘要、重要记忆、关键词更新
- CRUD 操作：增删改查记忆条目
- 摘要上限控制：超过上限时移除最早的非重要摘要
"""
import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from services.keyword_extractor import KeywordExtractor
from services.memory_index import MemoryIndex
from services.memory_retriever import MemoryRetriever
from services.memory_store import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)

# 延迟导入新模块的辅助函数，避免循环依赖
def _import_new_modules():
    """延迟导入新增模块，返回 (MemoryTagger, MemoryCompressor, ContextInjector, ActivePool)"""
    from services.memory_tagger import MemoryTagger
    from services.memory_compressor import MemoryCompressor
    from services.context_injector import ContextInjector
    from services.active_pool import ActivePool
    return MemoryTagger, MemoryCompressor, ContextInjector, ActivePool

# 默认配置
DEFAULT_MAX_SUMMARIES = 50  # QA 摘要数量上限
DEFAULT_KEYWORD_THRESHOLD = 3  # 关键词频率阈值
DEFAULT_RETRIEVAL_TOP_K = 3  # 记忆检索返回条数
QUESTION_MAX_LEN = 100  # 问题截取最大长度
ANSWER_MAX_LEN = 200  # 回答截取最大长度


class MemoryService:
    """记忆管理核心服务（单例）"""

    def __init__(self, data_dir: str, embedding_model_id: str = "local-minilm", use_sqlite: bool = False):
        """
        初始化记忆管理服务

        Args:
            data_dir: 记忆数据根目录，如 "data/memory/"
            embedding_model_id: embedding 模型 ID
            use_sqlite: 是否使用 SQLite 存储（可选增强）
        """
        self.data_dir = data_dir
        
        # 根据配置选择存储后端
        if use_sqlite:
            try:
                from services.memory_store_sqlite import MemoryStoreSQLite
                self.store = MemoryStoreSQLite(data_dir, use_sqlite=True)
                logger.info("使用 SQLite 存储后端（增强查询性能）")
            except Exception as e:
                logger.warning(f"SQLite 存储初始化失败，回退到 JSON: {e}")
                from services.memory_store import MemoryStore
                self.store = MemoryStore(data_dir)
        else:
            from services.memory_store import MemoryStore
            self.store = MemoryStore(data_dir)
        
        self.index = MemoryIndex(
            os.path.join(data_dir, "memory_index"), embedding_model_id
        )
        self.keyword_extractor = KeywordExtractor()
        self.max_summaries = DEFAULT_MAX_SUMMARIES
        self.keyword_threshold = DEFAULT_KEYWORD_THRESHOLD

        # 初始化新增模块（使用 _safe_execute 确保优雅降级）
        self.tagger = None
        self.compressor = None
        self.context_injector = None
        self.active_pool = None
        try:
            from config import settings as app_settings
        except Exception:
            app_settings = None

        try:
            MemoryTagger, MemoryCompressor, ContextInjector, ActivePool = _import_new_modules()
            self.tagger = MemoryTagger()
            compression_threshold = getattr(app_settings, "memory_compression_threshold", 20) if app_settings else 20
            self.compressor = MemoryCompressor(compression_threshold=compression_threshold)
            token_budget = getattr(app_settings, "memory_injection_token_budget", 800) if app_settings else 800
            self.context_injector = ContextInjector(token_budget=token_budget)
            pool_size = getattr(app_settings, "memory_active_pool_size", 100) if app_settings else 100
            self.active_pool = ActivePool(capacity=pool_size)
        except Exception as e:
            logger.warning(f"[MemoryService] 新增模块初始化失败，降级为基础功能: {e}")

        # 初始化检索器（传入 active_pool）
        self.retriever = MemoryRetriever(self.store, self.index, active_pool=self.active_pool)

        # 尝试加载已有的向量索引
        self.index.load()

        # 预加载活跃记忆池
        self._safe_execute("ActivePool.preload", self._preload_active_pool)

    # ==================== 安全执行与预加载 ====================

    def _safe_execute(self, component_name: str, func, *args, **kwargs):
        """安全执行组件方法，异常时降级

        Args:
            component_name: 组件名称（用于日志）
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值，异常时返回 None
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[{component_name}] 执行失败，降级处理: {e}")
            return None

    def _preload_active_pool(self) -> None:
        """预加载活跃记忆池（服务启动时调用）

        从存储中加载最近使用的记忆条目到 Active_Pool。
        """
        if not self.active_pool:
            return
        try:
            all_entries = self.store.get_all_entries()
            # 按 last_hit_at 降序排列，取前 N 条
            sorted_entries = sorted(
                all_entries,
                key=lambda e: e.last_hit_at or "",
                reverse=True,
            )
            self.active_pool.preload(sorted_entries[:self.active_pool.capacity])
        except Exception as e:
            logger.warning(f"[ActivePool] 预加载失败: {e}")

    # ==================== 记忆检索 ====================

    def retrieve_memories(
        self, query: str, top_k: int = DEFAULT_RETRIEVAL_TOP_K, api_key: str = None,
        doc_id: str = None, filter_by_doc: bool = False
    ) -> str:
        """检索相关记忆并返回格式化的上下文字符串

        Args:
            query: 用户查询文本
            top_k: 返回的最大结果数
            api_key: API 密钥（远程模型需要）
            doc_id: 当前文档 ID，用于文档相关性加权（可选）
            filter_by_doc: 是否只返回当前文档的记忆，默认 False（仅加权）

        Returns:
            格式化的记忆上下文字符串，无记忆时返回空字符串
        """
        try:
            # 定期评估记忆重要性（每 10 次检索评估一次，避免频繁计算）
            import random
            if random.random() < 0.1:  # 10% 概率触发评估
                try:
                    self.evaluate_and_update_importance()
                except Exception as e:
                    logger.debug(f"定期重要性评估失败（不影响检索）: {e}")
            
            memories = self.retriever.retrieve(
                query, top_k=top_k, api_key=api_key, 
                doc_id=doc_id, filter_by_doc=filter_by_doc
            )
            
            # 检索后触发晋升检查
            for mem in memories:
                try:
                    entry = MemoryEntry.from_dict(mem) if isinstance(mem, dict) else mem
                    self.check_and_promote(entry)
                except Exception as e:
                    logger.debug(f"晋升检查失败（不影响检索）: {e}")

            # 检索命中的记忆加入 Active_Pool（Page-In）
            if self.active_pool and memories:
                for mem in memories:
                    try:
                        entry = MemoryEntry.from_dict(mem) if isinstance(mem, dict) else mem
                        self.active_pool.put(entry)
                    except Exception as e:
                        logger.debug(f"Active_Pool Page-In 失败: {e}")
            
            return self.retriever.build_memory_context(memories)
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return ""

    def retrieve_memories_raw(
        self, query: str, top_k: int = DEFAULT_RETRIEVAL_TOP_K, api_key: str = None,
        doc_id: str = None, filter_by_doc: bool = False
    ) -> list[dict]:
        """检索相关记忆并返回原始记忆列表（供 ContextInjector 使用）

        Args:
            query: 用户查询文本
            top_k: 返回的最大结果数
            api_key: API 密钥
            doc_id: 当前文档 ID
            filter_by_doc: 是否只返回当前文档的记忆

        Returns:
            记忆字典列表，每条包含 content, memory_tier, importance 等字段
        """
        try:
            memories = self.retriever.retrieve(
                query, top_k=top_k, api_key=api_key,
                doc_id=doc_id, filter_by_doc=filter_by_doc
            )
            # 补充 memory_tier 和 tags 信息
            all_entries = self.store.get_all_entries()
            entry_map = {e.id: e for e in all_entries}
            enriched = []
            for mem in memories:
                entry_id = mem.get("entry_id", "")
                entry = entry_map.get(entry_id)
                enriched_mem = dict(mem)
                if entry:
                    enriched_mem["content"] = mem.get("text", entry.content)
                    enriched_mem["memory_tier"] = entry.memory_tier
                    enriched_mem["importance"] = entry.importance
                    enriched_mem["tags"] = entry.tags
                else:
                    enriched_mem.setdefault("content", mem.get("text", ""))
                    enriched_mem.setdefault("memory_tier", "short_term")
                    enriched_mem.setdefault("importance", 0.5)
                    enriched_mem.setdefault("tags", [])
                enriched.append(enriched_mem)
            return enriched
        except Exception as e:
            logger.error(f"记忆原始检索失败: {e}")
            return []

    # ==================== 分层记忆架构 ====================

    def check_and_promote(self, entry: MemoryEntry) -> None:
        """检查并执行记忆晋升

        晋升条件：
        - memory_tier 为 "short_term"
        - hit_count >= 晋升阈值（默认 5）
        - last_hit_at 在最近 7 天内

        Args:
            entry: 待检查的记忆条目
        """
        if entry.memory_tier != "short_term":
            return

        # 读取配置的晋升阈值
        try:
            from config import settings
            promotion_threshold = settings.memory_promotion_threshold
        except Exception:
            promotion_threshold = 5

        if entry.hit_count < promotion_threshold:
            return

        # 检查 last_hit_at 是否在最近 7 天内
        if not entry.last_hit_at:
            return

        try:
            last_hit_time = datetime.fromisoformat(entry.last_hit_at)
            if last_hit_time.tzinfo is None:
                last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return

        now = datetime.now(timezone.utc)
        if (now - last_hit_time) > timedelta(days=7):
            return

        # 满足晋升条件，更新 memory_tier
        entry.memory_tier = "long_term"
        self._persist_tier_change(entry)
        logger.info(f"记忆晋升: {entry.id} short_term -> long_term (hit_count={entry.hit_count})")

    def check_and_demote(self) -> None:
        """检查并执行记忆降级

        降级条件：
        - memory_tier 为 "long_term"
        - 距离 last_hit_at 超过降级天数（默认 90 天）
        - hit_count < 3
        """
        try:
            from config import settings
            demotion_days = settings.memory_demotion_days
        except Exception:
            demotion_days = 90

        now = datetime.now(timezone.utc)
        all_entries = self.store.get_all_entries()
        demoted_count = 0

        for entry in all_entries:
            if entry.memory_tier != "long_term":
                continue
            if entry.hit_count >= 3:
                continue

            # 计算距离 last_hit_at 的天数
            last_hit_time = None
            if entry.last_hit_at:
                try:
                    last_hit_time = datetime.fromisoformat(entry.last_hit_at)
                    if last_hit_time.tzinfo is None:
                        last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            if not last_hit_time:
                # 没有命中记录，使用创建时间
                try:
                    last_hit_time = datetime.fromisoformat(entry.created_at)
                    if last_hit_time.tzinfo is None:
                        last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

            days_since_hit = (now - last_hit_time).total_seconds() / 86400.0
            if days_since_hit > demotion_days:
                entry.memory_tier = "archived"
                self._persist_tier_change(entry)
                demoted_count += 1

        if demoted_count > 0:
            logger.info(f"记忆降级: {demoted_count} 条记忆 long_term -> archived")

    def _persist_tier_change(self, entry: MemoryEntry) -> None:
        """持久化记忆层级变更到存储

        在 profile 的 entries 和 session 的 important_memories 中查找并更新 memory_tier。

        Args:
            entry: 已更新 memory_tier 的记忆条目
        """
        # 在 profile 中查找
        profile = self.store.load_profile()
        for item in profile.get("entries", []):
            if item.get("id") == entry.id:
                item["memory_tier"] = entry.memory_tier
                self.store.save_profile(profile)
                return

        # 在 session 中查找
        if entry.doc_id:
            session = self.store.load_session(entry.doc_id)
            for item in session.get("important_memories", []):
                if item.get("id") == entry.id:
                    item["memory_tier"] = entry.memory_tier
                    self.store.save_session(entry.doc_id, session)
                    return

    def get_working_memory(self, chat_history: list[dict], window_size: int = None) -> list[dict]:
        """获取工作记忆（滑动窗口）

        从对话历史中提取最近 N 轮对话（user + assistant 配对）。

        Args:
            chat_history: 完整对话历史列表，每项包含 role 和 content
            window_size: 窗口大小（保留的轮数），默认从配置读取

        Returns:
            最近 N 轮对话的消息列表
        """
        if window_size is None:
            try:
                from config import settings
                window_size = settings.memory_working_window_size
            except Exception:
                window_size = 10

        if not chat_history:
            return []

        # 提取 user+assistant 配对的轮次
        rounds: list[list[dict]] = []
        i = 0
        while i < len(chat_history):
            msg = chat_history[i]
            if msg.get("role") == "user":
                # 尝试配对下一条 assistant 消息
                round_msgs = [msg]
                if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "assistant":
                    round_msgs.append(chat_history[i + 1])
                    i += 2
                else:
                    i += 1
                rounds.append(round_msgs)
            else:
                i += 1

        # 取最后 N 轮
        recent_rounds = rounds[-window_size:] if len(rounds) > window_size else rounds

        # 展平为消息列表
        result = []
        for round_msgs in recent_rounds:
            result.extend(round_msgs)
        return result

    # ==================== 记忆写入 ====================

    def save_qa_summary(
        self,
        doc_id: str,
        chat_history: list[dict],
        n: int = 3,
        api_key: str = None,
        model: str = None,
        api_provider: str = None,
    ) -> None:
        """从对话历史中提取最后 N 轮 QA 摘要并保存

        优先使用 LLM 提炼持久性事实（借鉴 OpenClaw），
        LLM 不可用时降级为截断摘要。

        Args:
            doc_id: 文档标识
            chat_history: 对话历史列表，每项包含 role 和 content
            n: 提取最后 N 轮 QA 对，默认 3
            api_key: LLM API 密钥（用于记忆提炼）
            model: LLM 模型名称
            api_provider: LLM 提供商
        """
        if not chat_history or not doc_id:
            return

        # 提取 QA 对：从对话历史中配对 user/assistant 消息
        qa_pairs = self._extract_qa_pairs(chat_history)
        if not qa_pairs:
            return

        # 取最后 N 轮
        recent_pairs = qa_pairs[-n:]

        # 加载当前 session
        session = self.store.load_session(doc_id)

        # 尝试 LLM 提炼
        distilled_facts = None
        if api_key and model and api_provider:
            distilled_facts = self._distill_facts(
                recent_pairs, api_key, model, api_provider
            )

        if distilled_facts:
            # LLM 提炼成功：每条事实作为一个高质量摘要
            for fact in distilled_facts:
                summary = {
                    "id": str(uuid.uuid4()),
                    "question": fact,
                    "answer": "",
                    "source_type": "llm_distilled",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "importance": 0.7,
                }
                session["qa_summaries"].append(summary)
            logger.info(f"LLM 记忆提炼: {len(distilled_facts)} 条事实")
        else:
            # 降级：截断摘要
            for question, answer in recent_pairs:
                truncated_q = question[:QUESTION_MAX_LEN]
                truncated_a = answer[:ANSWER_MAX_LEN]

                summary = {
                    "id": str(uuid.uuid4()),
                    "question": truncated_q,
                    "answer": truncated_a,
                    "source_type": "auto_qa",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "importance": 0.5,
                }
                session["qa_summaries"].append(summary)

        # 摘要数量上限控制
        self._enforce_summary_limit(session)

        # 更新最后访问时间并保存
        session["last_accessed"] = datetime.now(timezone.utc).isoformat()
        self.store.save_session(doc_id, session)
        
        # 同步写入 Markdown 源文件（每日日志）
        try:
            from services.memory_store import MemoryEntry
            for summary in session["qa_summaries"][-len(recent_pairs):]:
                # 只写入新添加的摘要
                entry = MemoryEntry(
                    id=summary.get("id", ""),
                    content=f"Q: {summary.get('question', '')}\nA: {summary.get('answer', '')}",
                    source_type=summary.get("source_type", "auto_qa"),
                    created_at=summary.get("created_at", ""),
                    doc_id=doc_id,
                    importance=summary.get("importance", 0.5),
                )
                self.store._write_memory_markdown(entry, is_long_term=False)
        except Exception as e:
            logger.warning(f"同步写入 Markdown 失败: {e}")

        # 保存完成后检查是否需要压缩（安全执行）
        self._safe_execute("MemoryCompressor.check", self._check_and_compress, doc_id, api_key, model, api_provider)

    def _check_and_compress(self, doc_id: str, api_key: str = None, model: str = None, api_provider: str = None) -> None:
        """检查并执行记忆压缩

        当同一文档的记忆条目数量超过压缩阈值时，触发压缩流程。

        Args:
            doc_id: 文档 ID
            api_key: LLM API 密钥
            model: LLM 模型名称
            api_provider: LLM 提供商
        """
        if not self.compressor or not doc_id:
            return
        all_entries = self.store.get_all_entries()
        if not self.compressor.should_compress(doc_id, all_entries):
            return
        # 筛选该文档的记忆条目
        doc_entries = [e for e in all_entries if e.doc_id == doc_id]
        compressed = self.compressor.compress(doc_entries, api_key=api_key, model=model, api_provider=api_provider)
        if compressed:
            # 删除原始条目，写入压缩后的条目
            for e in doc_entries:
                self.store.delete_entry(e.id)
            for c in compressed:
                self.store.add_entry(c)
            logger.info(f"[MemoryCompressor] 文档 {doc_id} 压缩完成: {len(doc_entries)} -> {len(compressed)}")

    def _distill_facts(
        self,
        qa_pairs: list[tuple[str, str]],
        api_key: str,
        model: str,
        api_provider: str,
    ) -> Optional[list[str]]:
        """使用 LLM 从 QA 对中提炼持久性事实

        借鉴 OpenClaw 记忆策略：让 LLM 决定哪些信息值得长期记住。

        Args:
            qa_pairs: [(question, answer), ...]
            api_key: API 密钥
            model: 模型名称
            api_provider: 提供商

        Returns:
            事实列表，失败时返回 None
        """
        try:
            from services.chat_service import call_ai_api

            # 构建 QA 文本
            qa_text = ""
            for q, a in qa_pairs:
                qa_text += f"用户问：{q[:500]}\nAI答：{a[:800]}\n\n"

            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是记忆提炼助手。从以下问答记录中提取值得长期记住的关键事实，"
                        "包括：用户偏好、重要结论、关键数据、用户纠正的错误。\n"
                        "规则：\n"
                        "- 每条事实一行，前面加 '- '\n"
                        "- 最多提取 5 条最重要的事实\n"
                        "- 只提取持久性信息，忽略临时性对话内容\n"
                        "- 如果没有值得记住的内容，只回复：无\n"
                        "- 不要添加任何解释或前缀"
                    ),
                },
                {"role": "user", "content": qa_text.strip()},
            ]

            # 在新事件循环中同步调用异步 LLM API
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                # 已在异步上下文中（不应该，因为是 threading 调用），用 run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    call_ai_api(
                        messages=messages,
                        api_key=api_key,
                        model=model,
                        provider=api_provider,
                        temperature=0.1,
                        max_tokens=300,
                    ),
                    loop,
                )
                response = future.result(timeout=30)
            else:
                response = asyncio.run(
                    call_ai_api(
                        messages=messages,
                        api_key=api_key,
                        model=model,
                        provider=api_provider,
                        temperature=0.1,
                        max_tokens=300,
                    )
                )

            if isinstance(response, dict) and response.get("error"):
                logger.warning(f"LLM 记忆提炼返回错误: {response['error']}")
                return None

            # 解析响应
            content = ""
            if isinstance(response, dict):
                content = response.get("content", "") or response.get("message", {}).get("content", "")
            elif isinstance(response, str):
                content = response

            if not content or content.strip() == "无":
                return None

            # 提取事实行
            facts = []
            for line in content.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    fact = line[2:].strip()
                    if fact and len(fact) > 3:
                        facts.append(fact)

            return facts if facts else None

        except Exception as e:
            logger.warning(f"LLM 记忆提炼失败，降级为截断: {e}")
            return None

    def _extract_qa_pairs(self, chat_history: list[dict]) -> list[tuple[str, str]]:
        """从对话历史中提取 QA 对

        遍历消息列表，将相邻的 user/assistant 消息配对。

        Args:
            chat_history: 对话历史列表

        Returns:
            [(question, answer), ...] 列表
        """
        pairs = []
        i = 0
        while i < len(chat_history) - 1:
            current = chat_history[i]
            next_msg = chat_history[i + 1]

            if (
                current.get("role") == "user"
                and next_msg.get("role") == "assistant"
            ):
                question = current.get("content", "")
                answer = next_msg.get("content", "")
                pairs.append((question, answer))
                i += 2  # 跳过已配对的两条消息
            else:
                i += 1

        return pairs

    def _enforce_summary_limit(self, session: dict) -> None:
        """摘要数量上限控制

        超过上限时移除最早的非重要摘要（importance < 1.0）。

        Args:
            session: 文档会话记忆字典
        """
        summaries = session.get("qa_summaries", [])
        while len(summaries) > self.max_summaries:
            # 查找最早的非重要摘要
            removed = False
            for i, s in enumerate(summaries):
                if s.get("importance", 0.5) < 1.0:
                    summaries.pop(i)
                    removed = True
                    break
            if not removed:
                # 所有摘要都是重要的，无法移除，退出循环
                break
        session["qa_summaries"] = summaries

    def save_important_memory(
        self,
        doc_id: str,
        question: str,
        answer: str,
        source_type: str = "manual",
    ) -> MemoryEntry:
        """保存重要记忆（用户手动标记或点赞）

        同时添加到 store 和 index（支持向量检索）。

        Args:
            doc_id: 文档标识
            question: 用户问题
            answer: AI 回答
            source_type: 来源类型，"manual" 或 "liked"

        Returns:
            创建的 MemoryEntry 对象
        """
        content = f"Q: {question}\nA: {answer}"
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            source_type=source_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            doc_id=doc_id,
            importance=1.0,  # 重要记忆默认 1.0
            memory_tier="long_term",  # 手动标记直接进入长期记忆
        )

        # 保存到 session 的 important_memories
        session = self.store.load_session(doc_id)
        session["important_memories"].append(entry.to_dict())
        session["last_accessed"] = datetime.now(timezone.utc).isoformat()
        self.store.save_session(doc_id, session)
        
        # 同步写入 Markdown 源文件（每日日志）
        try:
            self.store._write_memory_markdown(entry, is_long_term=False)
        except Exception as e:
            logger.warning(f"同步写入 Markdown 失败: {e}")

        # 添加到向量索引
        try:
            self.index.add_entry(entry.id, content)
        except Exception as e:
            logger.error(f"添加重要记忆到向量索引失败: {e}")

        return entry

    def update_keywords(self, query: str) -> None:
        """从查询中提取关键词并更新用户画像

        提取关键词 → 更新频率统计 → 更新关注领域列表。

        Args:
            query: 用户查询文本
        """
        if not query or not query.strip():
            return

        keywords = self.keyword_extractor.extract_keywords(query)
        if not keywords:
            return

        profile = self.store.load_profile()
        profile = self.keyword_extractor.update_frequency(profile, keywords)

        # 更新关注领域列表
        profile["focus_areas"] = self.keyword_extractor.get_focus_areas(
            profile, threshold=self.keyword_threshold
        )

        self.store.save_profile(profile)

    # ==================== CRUD 操作 ====================

    def get_profile(self) -> dict:
        """获取用户画像数据"""
        return self.store.load_profile()

    def get_session(self, doc_id: str) -> dict:
        """获取指定文档的会话记忆"""
        return self.store.load_session(doc_id)

    def add_entry(
        self, content: str, source_type: str, doc_id: str = None
    ) -> MemoryEntry:
        """添加记忆条目

        同时添加到 store 和 index，并自动打标签。

        Args:
            content: 记忆内容文本
            source_type: 来源类型
            doc_id: 关联的文档 ID（可选）

        Returns:
            创建的 MemoryEntry 对象
        """
        importance = 1.0 if source_type in ("manual", "liked") else 0.5
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            source_type=source_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            doc_id=doc_id,
            importance=importance,
        )

        # 自动打标签（安全执行，失败不影响写入）
        if self.tagger:
            tags = self._safe_execute("MemoryTagger.auto_tag", self.tagger.auto_tag, content)
            if tags:
                entry.tags = tags

        # 保存到 store
        self.store.add_entry(entry)

        # 添加到向量索引
        try:
            self.index.add_entry(entry.id, content)
        except Exception as e:
            logger.error(f"添加记忆条目到向量索引失败: {e}")

        return entry

    def delete_entry(self, entry_id: str) -> bool:
        """删除指定记忆条目

        同时从 store 和 index 中移除。

        Args:
            entry_id: 记忆条目 ID

        Returns:
            是否删除成功
        """
        success = self.store.delete_entry(entry_id)
        if success:
            try:
                self.index.remove_entry(entry_id)
            except Exception as e:
                logger.error(f"从向量索引移除记忆条目失败: {e}")
        return success

    def update_entry(self, entry_id: str, content: str) -> bool:
        """更新指定记忆条目的内容

        同时更新 store 和 index。

        Args:
            entry_id: 记忆条目 ID
            content: 新的内容文本

        Returns:
            是否更新成功
        """
        success = self.store.update_entry(entry_id, content)
        if success:
            try:
                # 先移除旧的向量，再添加新的
                self.index.remove_entry(entry_id)
                self.index.add_entry(entry_id, content)
            except Exception as e:
                logger.error(f"更新向量索引失败: {e}")
        return success

    def clear_all(self) -> None:
        """清空所有记忆数据

        同时清空 store 和 index。
        """
        self.store.clear_all()
        try:
            self.index.rebuild([])
        except Exception as e:
            logger.error(f"清空向量索引失败: {e}")

    def evaluate_and_update_importance(self) -> None:
        """自动评估并更新记忆重要性
        
        基于以下因素综合评分：
        - 命中次数：频繁使用的记忆提升重要性
        - 时间衰减：长期未使用的记忆降低重要性
        - 用户标记：手动标记的记忆保持高重要性
        
        自动升降级规则：
        - 命中次数 >= 5 且最近 7 天内使用过：提升到 0.8
        - 命中次数 >= 10 且最近 3 天内使用过：提升到 1.0
        - 超过 90 天未使用且命中次数 < 3：降低到 0.3
        - 超过 180 天未使用：降低到 0.1
        """
        try:
            from datetime import timedelta
            
            all_entries = self.store.get_all_entries()
            now = datetime.now(timezone.utc)
            updated_count = 0
            
            for entry in all_entries:
                original_importance = entry.importance
                new_importance = original_importance
                
                # 用户标记的记忆（importance >= 1.0）保持不变
                if entry.importance >= 1.0:
                    continue
                
                # 计算最后使用时间
                last_hit_time = None
                if entry.last_hit_at:
                    try:
                        last_hit_time = datetime.fromisoformat(entry.last_hit_at)
                        if last_hit_time.tzinfo is None:
                            last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        pass
                
                if not last_hit_time:
                    # 没有命中记录，使用创建时间
                    try:
                        last_hit_time = datetime.fromisoformat(entry.created_at)
                        if last_hit_time.tzinfo is None:
                            last_hit_time = last_hit_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue
                
                days_since_use = (now - last_hit_time).total_seconds() / 86400.0
                hit_count = entry.hit_count
                
                # 自动提升规则
                if hit_count >= 10 and days_since_use <= 3:
                    new_importance = 1.0  # 最高重要性
                elif hit_count >= 5 and days_since_use <= 7:
                    new_importance = max(original_importance, 0.8)  # 提升到 0.8
                elif hit_count >= 3 and days_since_use <= 14:
                    new_importance = max(original_importance, 0.6)  # 提升到 0.6
                
                # 自动降级规则
                elif days_since_use > 180:
                    new_importance = 0.1  # 最低重要性
                elif days_since_use > 90 and hit_count < 3:
                    new_importance = min(original_importance, 0.3)  # 降低到 0.3
                elif days_since_use > 60 and hit_count < 2:
                    new_importance = min(original_importance, 0.4)  # 降低到 0.4
                
                # 如果重要性发生变化，更新条目
                if abs(new_importance - original_importance) > 0.05:  # 变化超过 5% 才更新
                    entry.importance = new_importance
                    # 更新存储
                    if entry.doc_id:
                        session = self.store.load_session(entry.doc_id)
                        # 在 important_memories 中查找并更新
                        for item in session.get("important_memories", []):
                            if item.get("id") == entry.id:
                                item["importance"] = new_importance
                                self.store.save_session(entry.doc_id, session)
                                updated_count += 1
                                break
                    else:
                        # 在 profile 中查找并更新
                        profile = self.store.load_profile()
                        for item in profile.get("entries", []):
                            if item.get("id") == entry.id:
                                item["importance"] = new_importance
                                self.store.save_profile(profile)
                                updated_count += 1
                                break
            
            if updated_count > 0:
                logger.info(f"自动评估记忆重要性: 更新了 {updated_count} 条记忆的重要性")
            
            # 评估完成后触发降级检查
            try:
                self.check_and_demote()
            except Exception as e:
                logger.debug(f"降级检查失败（不影响评估）: {e}")
        except Exception as e:
            logger.error(f"自动评估记忆重要性失败: {e}")
    
    def get_status(self) -> dict:
        """获取记忆系统状态

        Returns:
            包含 enabled、total_entries、index_size、profile_focus_areas 的字典
        """
        try:
            all_entries = self.store.get_all_entries()
            total_entries = len(all_entries)
        except Exception:
            total_entries = 0

        index_size = (
            self.index.index.ntotal
            if self.index.index is not None
            else 0
        )

        profile = self.store.load_profile()
        focus_areas = profile.get("focus_areas", [])

        return {
            "enabled": True,
            "total_entries": total_entries,
            "index_size": index_size,
            "profile_focus_areas": focus_areas,
        }
