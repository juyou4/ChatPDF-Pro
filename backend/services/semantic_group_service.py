"""
语义意群服务模块

提供语义意群（Semantic Group）的数据结构定义和持久化功能。
语义意群是将多个文本分块聚合成的语义完整单元，每个意群包含
摘要（summary）、精要（digest）和全文（full_text）三种粒度的文本表示。

存储路径：Chatpdf/data/semantic_groups/{doc_id}.json
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# 当前数据格式版本号，用于数据格式演进
SCHEMA_VERSION = 1


@dataclass
class SemanticGroup:
    """语义意群数据结构

    将多个文本分块聚合成语义完整的单元（约 5000 字符），
    每个意群包含摘要、精要和全文三种粒度的文本表示。

    Attributes:
        group_id: 意群唯一标识，如 "group-0"
        chunk_indices: 包含的分块索引列表
        char_count: 总字符数
        summary: 摘要（≤80字）
        digest: 精要（≤1000字）
        full_text: 完整文本
        keywords: 关键词列表（4-6个）
        page_range: 页码范围 (start_page, end_page)
        summary_status: 摘要生成状态: "ok" | "failed" | "truncated"
        llm_meta: LLM 调用元数据: {model, temperature, prompt_version, created_at}
    """

    group_id: str
    chunk_indices: List[int]
    char_count: int
    summary: str
    digest: str
    full_text: str
    keywords: List[str]
    page_range: Tuple[int, int]
    summary_status: str = "ok"
    llm_meta: Optional[dict] = None

    def to_dict(self) -> dict:
        """将意群对象转换为可序列化的字典

        Returns:
            包含所有字段的字典，page_range 转换为列表格式
        """
        data = asdict(self)
        # page_range 是 tuple，JSON 序列化时转为 list，这里显式处理确保一致性
        data["page_range"] = list(self.page_range)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticGroup":
        """从字典创建意群对象

        Args:
            data: 包含意群字段的字典

        Returns:
            SemanticGroup 实例
        """
        # page_range 从 JSON 加载后是 list，需要转回 tuple
        page_range = data.get("page_range", [0, 0])
        if isinstance(page_range, list):
            page_range = tuple(page_range)

        return cls(
            group_id=data["group_id"],
            chunk_indices=data["chunk_indices"],
            char_count=data["char_count"],
            summary=data["summary"],
            digest=data["digest"],
            full_text=data["full_text"],
            keywords=data["keywords"],
            page_range=page_range,
            summary_status=data.get("summary_status", "ok"),
            llm_meta=data.get("llm_meta"),
        )


class SemanticGroupService:
    """语义意群服务

    负责语义意群的生成、持久化和加载。
    支持通过 LLM API 生成摘要和关键词，失败时自动降级为文本截断。

    Attributes:
        api_key: LLM API 密钥
        model: LLM 模型名称（默认 "gpt-4o-mini"）
        provider: LLM 提供商标识（默认 "openai"）
        endpoint: LLM API 端点 URL（可选，为空时使用提供商默认端点）
        temperature: LLM 生成温度（默认 0.3，较低以获得稳定输出）
        prompt_version: 提示词版本标识，用于追踪（默认 "v1"）
    """

    # 摘要生成提示词模板
    _SUMMARY_PROMPT_TEMPLATE = (
        "请对以下文本生成一段不超过{max_length}字的中文摘要。"
        "要求：简洁准确，保留核心信息，不要添加额外解释。"
        "只输出摘要内容，不要包含任何前缀或说明。\n\n"
        "文本内容：\n{text}"
    )

    # 关键词提取提示词模板
    _KEYWORDS_PROMPT_TEMPLATE = (
        "请从以下文本中提取4到6个最重要的关键词。"
        "要求：只返回关键词列表，每个关键词用逗号分隔，"
        "不要添加编号或额外解释。\n\n"
        "文本内容：\n{text}"
    )

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        endpoint: str = "",
        temperature: float = 0.3,
        prompt_version: str = "v1",
    ):
        """初始化语义意群服务

        Args:
            api_key: LLM API 密钥
            model: LLM 模型名称
            provider: LLM 提供商标识
            endpoint: LLM API 端点 URL
            temperature: LLM 生成温度
            prompt_version: 提示词版本标识
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.endpoint = endpoint
        self.temperature = temperature
        self.prompt_version = prompt_version

    def save_groups(
        self,
        doc_id: str,
        groups: List[SemanticGroup],
        store_dir: str,
        doc_hash: str = "",
        config: Optional[dict] = None,
    ) -> None:
        """将意群数据序列化为 JSON 并保存到磁盘

        JSON 格式包含 schema_version 字段用于数据格式演进。

        Args:
            doc_id: 文档唯一标识
            groups: 语义意群列表
            store_dir: 存储目录路径
            doc_hash: 文档哈希值（可选），如 "sha256:..."
            config: 聚合配置参数（可选），如 {target_chars, min_chars, max_chars}
        """
        # 构建持久化数据结构
        data = {
            "schema_version": SCHEMA_VERSION,
            "doc_id": doc_id,
            "doc_hash": doc_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config or {
                "target_chars": 5000,
                "min_chars": 2500,
                "max_chars": 6000,
            },
            "groups": [group.to_dict() for group in groups],
        }

        # 确保存储目录存在
        os.makedirs(store_dir, exist_ok=True)

        file_path = os.path.join(store_dir, f"{doc_id}.json")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"意群数据已保存: {file_path}，共 {len(groups)} 个意群")
        except Exception as e:
            logger.error(f"保存意群数据失败: {file_path}，错误: {e}")
            raise

    def load_groups(
        self, doc_id: str, store_dir: str
    ) -> Optional[List[SemanticGroup]]:
        """从磁盘加载意群数据，解析 JSON 并还原为意群对象列表

        如果文件不存在、格式无效或 schema_version 不匹配，返回 None。

        Args:
            doc_id: 文档唯一标识
            store_dir: 存储目录路径

        Returns:
            语义意群列表，加载失败时返回 None
        """
        file_path = os.path.join(store_dir, f"{doc_id}.json")

        if not os.path.exists(file_path):
            logger.info(f"意群数据文件不存在: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"意群数据文件损坏或格式无效: {file_path}，错误: {e}")
            return None
        except Exception as e:
            logger.error(f"读取意群数据文件失败: {file_path}，错误: {e}")
            return None

        # 校验 schema_version
        schema_version = data.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            logger.info(
                f"意群数据 schema_version 不匹配: "
                f"文件版本={schema_version}，当前版本={SCHEMA_VERSION}，"
                f"需要重新生成意群"
            )
            return None

        # 校验 groups 字段存在且为列表
        groups_data = data.get("groups")
        if not isinstance(groups_data, list):
            logger.error(f"意群数据格式无效: groups 字段缺失或类型错误: {file_path}")
            return None

        # 反序列化每个意群
        try:
            groups = [SemanticGroup.from_dict(g) for g in groups_data]
            logger.info(f"意群数据已加载: {file_path}，共 {len(groups)} 个意群")
            return groups
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"反序列化意群数据失败: {file_path}，错误: {e}")
            return None

    # ---- 以下方法将在后续任务中实现 ----

    # ---- 标题检测正则表达式 ----
    # 编号模式：如 "1." "1.1" "1.1.1" "2.3.4" 等（行首可有空白）
    _RE_NUMBERED_HEADING = re.compile(r"^\s*\d+(\.\d+)*\.?\s+\S")
    # Markdown 标题：如 "# 标题" "## 标题" 等
    _RE_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
    # 表格行：包含 | 分隔符的行（至少两个 |）
    _RE_TABLE_ROW = re.compile(r"^.*\|.*\|")
    # 代码块标记：``` 开头
    _RE_CODE_FENCE = re.compile(r"^\s*```")

    def _is_heading_line(self, line: str) -> bool:
        """判断一行文本是否为标题行

        标题判定规则：
        - 编号模式：如 "1." "1.1" "2.3.4" 开头
        - 全大写行：长度 >= 2 且全部为大写字母（忽略空白和标点）
        - Markdown # 标题

        Args:
            line: 待检测的文本行

        Returns:
            是否为标题行
        """
        stripped = line.strip()
        if not stripped:
            return False

        # 编号模式
        if self._RE_NUMBERED_HEADING.match(stripped):
            return True

        # Markdown 标题
        if self._RE_MARKDOWN_HEADING.match(stripped):
            return True

        # 全大写行：提取字母字符，判断是否全部大写且长度 >= 2
        alpha_chars = re.sub(r"[^a-zA-Z]", "", stripped)
        if len(alpha_chars) >= 2 and alpha_chars.isupper():
            return True

        return False

    def _is_table_or_code_boundary(self, line: str) -> bool:
        """判断一行文本是否为表格行或代码块标记

        Args:
            line: 待检测的文本行

        Returns:
            是否为表格/代码块边界
        """
        stripped = line.strip()
        if not stripped:
            return False

        # 代码块标记
        if self._RE_CODE_FENCE.match(stripped):
            return True

        # 表格行
        if self._RE_TABLE_ROW.match(stripped):
            return True

        return False

    def _detect_boundary(
        self, chunk: str, next_chunk: str, page: int, next_page: int
    ) -> bool:
        """检测两个相邻分块之间是否存在硬边界

        硬边界规则：
        1. 不跨页面边界（page != next_page 时强制切分）
        2. 不跨标题边界（下一个分块以标题行开头时切分）
        3. 不跨表格/代码块边界（下一个分块以表格行或代码块标记开头时切分）

        Args:
            chunk: 当前分块文本
            next_chunk: 下一个分块文本
            page: 当前分块所在页码
            next_page: 下一个分块所在页码

        Returns:
            True 表示存在硬边界，应在此处切分
        """
        # 规则 1：页面边界
        if page != next_page:
            return True

        # 获取下一个分块的第一行（用于检测标题和表格/代码块边界）
        next_first_line = next_chunk.split("\n", 1)[0] if next_chunk else ""

        # 规则 2：标题边界
        if self._is_heading_line(next_first_line):
            return True

        # 规则 3：表格/代码块边界
        if self._is_table_or_code_boundary(next_first_line):
            return True

        return False

    def _aggregate_chunks(
        self,
        chunks: List[str],
        chunk_pages: List[int],
        target_chars: int = 5000,
        min_chars: int = 2500,
        max_chars: int = 6000,
    ) -> List[dict]:
        """按字符数阈值将连续分块聚合为候选意群

        聚合逻辑：
        1. 依次遍历分块，累计字符数
        2. 当累计字符数达到 target_chars 或遇到硬边界时，切分为一个候选意群
        3. 当累计字符数超过 max_chars 时强制切分
        4. 最后剩余的分块组成最后一个候选意群

        Args:
            chunks: 文本分块列表
            chunk_pages: 每个分块对应的页码列表
            target_chars: 目标字符数（默认 5000）
            min_chars: 最小字符数（默认 2500）
            max_chars: 最大字符数（默认 6000）

        Returns:
            候选意群列表，每个元素为字典：
            {
                "chunk_indices": List[int],  # 包含的分块索引
                "full_text": str,            # 拼接后的完整文本
                "char_count": int,           # 总字符数
                "page_range": Tuple[int, int]  # 页码范围 (start, end)
            }
        """
        if not chunks:
            return []

        candidates = []  # 最终的候选意群列表

        # 当前正在累积的意群状态
        current_indices = []
        current_texts = []
        current_char_count = 0

        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk)

            # 检查是否需要在当前分块之前切分（硬边界检测）
            if current_indices:
                # 检测与前一个分块之间是否存在硬边界
                has_boundary = self._detect_boundary(
                    chunks[i - 1], chunk, chunk_pages[i - 1], chunk_pages[i]
                )

                # 切分条件：
                # 1. 遇到硬边界
                # 2. 加入当前分块后超过 max_chars
                # 3. 已达到 target_chars（且已超过 min_chars）
                should_split = False

                if has_boundary:
                    # 硬边界：无论字符数多少都切分
                    should_split = True
                elif current_char_count + chunk_len > max_chars:
                    # 超过最大字符数：强制切分
                    should_split = True
                elif current_char_count >= target_chars:
                    # 已达到目标字符数：切分
                    should_split = True

                if should_split:
                    # 将当前累积的分块保存为一个候选意群
                    full_text = "\n".join(current_texts)
                    pages = [chunk_pages[idx] for idx in current_indices]
                    candidates.append({
                        "chunk_indices": list(current_indices),
                        "full_text": full_text,
                        "char_count": len(full_text),
                        "page_range": (min(pages), max(pages)),
                    })
                    # 重置累积状态
                    current_indices = []
                    current_texts = []
                    current_char_count = 0

            # 将当前分块加入累积
            current_indices.append(i)
            current_texts.append(chunk)
            current_char_count += chunk_len

        # 处理最后剩余的分块
        if current_indices:
            full_text = "\n".join(current_texts)
            pages = [chunk_pages[idx] for idx in current_indices]
            candidates.append({
                "chunk_indices": list(current_indices),
                "full_text": full_text,
                "char_count": len(full_text),
                "page_range": (min(pages), max(pages)),
            })

        return candidates

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM API 获取文本响应

        使用 httpx 直接调用 OpenAI 兼容的 chat completions API。
        支持通过 provider 和 endpoint 配置不同的 LLM 提供商。

        Args:
            prompt: 用户提示词

        Returns:
            LLM 生成的文本内容

        Raises:
            Exception: API 调用失败时抛出异常
        """
        from models.provider_registry import PROVIDER_CONFIG

        # 确定 API 端点
        endpoint = self.endpoint
        if not endpoint:
            provider_cfg = PROVIDER_CONFIG.get(self.provider, {})
            endpoint = provider_cfg.get("endpoint", "https://api.openai.com/v1/chat/completions")

        messages = [
            {"role": "user", "content": prompt}
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2000,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            if response.status_code != 200:
                raise Exception(
                    f"LLM API 调用失败: HTTP {response.status_code}, {response.text}"
                )
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content.strip()

    def _build_llm_meta(self) -> dict:
        """构建 LLM 调用元数据

        Returns:
            包含 model、temperature、prompt_version、created_at 的字典
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "prompt_version": self.prompt_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _generate_summary(
        self, text: str, max_length: int
    ) -> Tuple[str, str]:
        """调用 LLM 生成摘要，返回 (summary, status)

        根据 max_length 参数生成不同粒度的摘要：
        - max_length=80：生成 summary（简短摘要）
        - max_length=1000：生成 digest（精要）

        失败时降级为文本截断，status 标记为 "failed"。

        Args:
            text: 待摘要的原始文本
            max_length: 摘要最大字符数（80 或 1000）

        Returns:
            (summary_text, status) 元组
            status 取值: "ok" | "failed"
        """
        # 如果原文本身就短于 max_length，直接返回原文
        if len(text) <= max_length:
            return text, "ok"

        # 如果没有配置 API key，直接降级为截断
        if not self.api_key:
            logger.warning("未配置 LLM API key，降级为文本截断")
            return text[:max_length], "failed"

        try:
            # 构建提示词，限制输入文本长度避免超出 LLM 上下文
            # 对于摘要生成，输入文本最多取前 8000 字符
            input_text = text[:8000] if len(text) > 8000 else text
            prompt = self._SUMMARY_PROMPT_TEMPLATE.format(
                max_length=max_length, text=input_text
            )

            result = await self._call_llm(prompt)

            # 确保结果不超过 max_length
            if len(result) > max_length:
                result = result[:max_length]

            if not result.strip():
                # LLM 返回空内容，降级为截断
                logger.warning("LLM 返回空摘要，降级为文本截断")
                return text[:max_length], "failed"

            return result, "ok"

        except Exception as e:
            logger.warning(f"LLM 摘要生成失败，降级为文本截断: {e}")
            return text[:max_length], "failed"

    async def _extract_keywords(self, text: str) -> List[str]:
        """调用 LLM 提取关键词，失败时返回空列表

        从文本中提取 4-6 个最重要的关键词。

        Args:
            text: 待提取关键词的文本

        Returns:
            关键词列表（4-6 个），失败时返回空列表
        """
        # 如果没有配置 API key，直接返回空列表
        if not self.api_key:
            logger.warning("未配置 LLM API key，无法提取关键词")
            return []

        try:
            # 限制输入文本长度
            input_text = text[:8000] if len(text) > 8000 else text
            prompt = self._KEYWORDS_PROMPT_TEMPLATE.format(text=input_text)

            result = await self._call_llm(prompt)

            if not result.strip():
                logger.warning("LLM 返回空关键词结果")
                return []

            # 解析关键词：按逗号、顿号或换行符分隔
            # 支持中文逗号、英文逗号、顿号等分隔符
            raw_keywords = re.split(r"[,，、\n]+", result.strip())
            # 清理每个关键词：去除空白、编号前缀（如 "1." "- "）
            keywords = []
            for kw in raw_keywords:
                kw = kw.strip()
                # 去除可能的编号前缀
                kw = re.sub(r"^\d+[\.\)、]\s*", "", kw)
                kw = re.sub(r"^[-•]\s*", "", kw)
                kw = kw.strip()
                if kw:
                    keywords.append(kw)

            # 限制关键词数量为 4-6 个
            if len(keywords) > 6:
                keywords = keywords[:6]

            return keywords

        except Exception as e:
            logger.warning(f"LLM 关键词提取失败: {e}")
            return []

    async def generate_groups(
        self,
        chunks: List[str],
        chunk_pages: List[int],
        target_chars: int = 5000,
        min_chars: int = 2500,
        max_chars: int = 6000,
    ) -> List[SemanticGroup]:
        """将分块聚合为语义意群

        流程：
        1. 调用 _aggregate_chunks 聚合分块为候选意群
        2. 对每个候选意群：
           - 调用 _generate_summary(text, 80) 生成 summary
           - 调用 _generate_summary(text, 1000) 生成 digest
           - 调用 _extract_keywords(text) 提取关键词
           - 构建 SemanticGroup 对象
        3. 返回完整的 SemanticGroup 列表

        Args:
            chunks: 文本分块列表
            chunk_pages: 每个分块对应的页码列表
            target_chars: 目标字符数（默认 5000）
            min_chars: 最小字符数（默认 2500）
            max_chars: 最大字符数（默认 6000）

        Returns:
            语义意群列表
        """
        # 步骤 1：聚合分块为候选意群
        candidates = self._aggregate_chunks(
            chunks, chunk_pages, target_chars, min_chars, max_chars
        )

        if not candidates:
            return []

        # 构建 LLM 元数据（所有意群共享同一次生成的元数据）
        llm_meta = self._build_llm_meta()

        groups: List[SemanticGroup] = []

        # 步骤 2：对每个候选意群生成摘要、精要和关键词
        for index, candidate in enumerate(candidates):
            full_text = candidate["full_text"]

            # 生成 summary（≤80 字）和 digest（≤1000 字）
            summary, summary_status = await self._generate_summary(full_text, 80)
            digest, digest_status = await self._generate_summary(full_text, 1000)

            # 提取关键词
            keywords = await self._extract_keywords(full_text)

            # summary_status 取 summary 和 digest 中较差的状态
            # 如果任一为 "failed" 则标记为 "failed"
            if summary_status == "failed" or digest_status == "failed":
                final_status = "failed"
            else:
                final_status = "ok"

            # 构建 SemanticGroup 对象
            group = SemanticGroup(
                group_id=f"group-{index}",
                chunk_indices=candidate["chunk_indices"],
                char_count=candidate["char_count"],
                summary=summary,
                digest=digest,
                full_text=full_text,
                keywords=keywords,
                page_range=candidate["page_range"],
                summary_status=final_status,
                llm_meta=llm_meta,
            )
            groups.append(group)

        logger.info(f"语义意群生成完成，共 {len(groups)} 个意群")
        return groups
