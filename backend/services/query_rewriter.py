"""
查询改写服务 - 使用本地规则将口语化查询转换为检索友好形式

不依赖 LLM 调用，使用正则表达式和模式匹配实现改写。
支持中英文混合查询。
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QueryRewriter:
    """查询改写器 - 使用本地规则将口语化查询转换为检索友好形式"""

    # 口语化模式 → 规范化替换的映射表
    # 每个元素为 (正则模式, 替换字符串) 的元组
    # 注意：顺序很重要，更具体的模式应放在前面
    COLLOQUIAL_PATTERNS: list[tuple[re.Pattern, str]] = [
        # "啥意思" 相关模式
        (re.compile(r'(.+?)啥意思'), r'\1的含义和解释'),
        (re.compile(r'啥意思'), r'的含义和解释'),

        # "讲了啥" / "说了啥" 相关模式
        (re.compile(r'(.+?)讲了啥'), r'\1的主要内容'),
        (re.compile(r'(.+?)说了啥'), r'\1的主要内容'),
        (re.compile(r'讲了啥'), r'的主要内容'),
        (re.compile(r'说了啥'), r'的主要内容'),

        # "啥是 X" / "X 是啥" 定义类模式
        (re.compile(r'啥是\s*(.+)'), r'\1的定义'),
        (re.compile(r'(.+?)是啥'), r'\1的定义'),

        # "为啥" 原因类模式
        (re.compile(r'为啥要?\s*(.+)'), r'\1的原因和目的'),
        (re.compile(r'为啥'), r'的原因'),

        # "咋用" / "怎么用" 使用方法类模式
        (re.compile(r'咋用\s*(.+)'), r'\1的使用方法'),
        (re.compile(r'(.+?)咋用'), r'\1的使用方法'),
        (re.compile(r'怎么用\s*(.+)'), r'\1的使用方法'),
        (re.compile(r'(.+?)怎么用'), r'\1的使用方法'),

        # "咋 X" → "如何 X" 通用模式
        (re.compile(r'咋(.+)'), r'如何\1'),

        # "啥 X" → "什么 X" 通用模式
        (re.compile(r'啥(.+)'), r'什么\1'),
    ]

    # 指示代词列表
    PRONOUN_PATTERNS: list[str] = [
        '这个', '那个', '这块', '那块', '这部分', '那部分',
        '这段', '那段', '这里', '那里', '它',
    ]

    def rewrite(self, query: str, selected_text: Optional[str] = None) -> str:
        """改写查询

        处理流程：
        1. 如果提供了 selected_text，先尝试解析指示代词
        2. 然后应用口语化模式替换
        3. 如果无需改写则返回原始查询

        Args:
            query: 原始用户查询
            selected_text: 用户选中的文本上下文（可选）

        Returns:
            改写后的查询文本，如果无需改写则返回原始查询
        """
        if not query or not query.strip():
            return query

        try:
            rewritten = query

            # 第一步：如果有选中文本，尝试解析指示代词
            if selected_text and selected_text.strip():
                rewritten = self._resolve_pronouns(rewritten, selected_text)

            # 第二步：应用口语化模式替换
            rewritten = self._replace_colloquial(rewritten)

            # 第三步（新增）：如果有 selected_text 且查询未被代词解析改变，
            # 追加 selected_text 关键内容以增强检索
            if selected_text and selected_text.strip() and rewritten == query:
                rewritten = self._augment_with_selected_text(rewritten, selected_text)

            return rewritten
        except Exception as e:
            # 改写失败时返回原始查询，记录警告日志
            logger.warning(f"查询改写失败，返回原始查询: {e}")
            return query

    def _replace_colloquial(self, query: str) -> str:
        """替换口语化表达

        遍历 COLLOQUIAL_PATTERNS，对查询应用正则替换。
        如果没有匹配到任何模式，返回原始查询。

        Args:
            query: 待处理的查询文本

        Returns:
            替换后的查询文本
        """
        result = query
        for pattern, replacement in self.COLLOQUIAL_PATTERNS:
            new_result = pattern.sub(replacement, result)
            if new_result != result:
                # 匹配到模式，使用替换结果
                result = new_result
                # 只应用第一个匹配的模式，避免多次替换导致语义混乱
                break
        return result

    def _resolve_pronouns(self, query: str, selected_text: str) -> str:
        """解析指示代词，用选中文本的关键内容替换

        检查查询中是否包含指示代词，如果包含则用选中文本的
        关键内容片段替换。

        Args:
            query: 包含指示代词的查询
            selected_text: 用户选中的文本上下文

        Returns:
            替换指示代词后的查询
        """
        if not selected_text or not selected_text.strip():
            return query

        # 提取选中文本的关键内容
        key_content = self._extract_key_content(selected_text)
        if not key_content:
            return query

        result = query
        for pronoun in self.PRONOUN_PATTERNS:
            if pronoun in result:
                # 用关键内容替换指示代词
                result = result.replace(pronoun, key_content, 1)

        return result

    def _augment_with_selected_text(
        self, query: str, selected_text: str, max_augment_chars: int = 80
    ) -> str:
        """将 selected_text 关键内容追加到查询中增强检索语义

        仅在查询未被指示代词解析改变时调用。
        提取 selected_text 的关键内容片段，追加到查询末尾。

        Args:
            query: 原始查询
            selected_text: 用户选中的文本
            max_augment_chars: 追加内容的最大字符数，默认 80

        Returns:
            追加关键内容后的查询，如果无法提取关键内容则返回原始查询
        """
        key_content = self._extract_key_content(selected_text, max_chars=max_augment_chars)
        if not key_content:
            return query
        return f"{query} {key_content}"

    async def rewrite_with_llm(
        self,
        query: str,
        chat_history: list[dict] = None,
        selected_text: Optional[str] = None,
        api_key: str = "",
        model: str = "",
        provider: str = "",
        endpoint: str = "",
    ) -> str:
        """用 LLM 改写查询，支持多轮对话指代消解

        先执行 regex 规则改写（零成本），再用 LLM 改写。
        适用于多轮对话中"那它呢？""具体怎么做？"等需要上下文的查询。

        Args:
            query: 原始查询
            chat_history: 对话历史 [{"role": "user"|"assistant", "content": "..."}]
            selected_text: 用户选中的文本
            api_key: LLM API 密钥
            model: LLM 模型名称
            provider: LLM 提供商
            endpoint: LLM API 端点

        Returns:
            改写后的查询，失败时返回 regex 改写结果
        """
        # 第一步：先跑 regex 改写
        rewritten = self.rewrite(query, selected_text=selected_text)

        if not api_key:
            return rewritten

        # 第二步：构建 LLM 改写 prompt
        history_text = ""
        if chat_history:
            recent = chat_history[-6:]  # 最近 3 轮对话
            history_lines = []
            for msg in recent:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")[:200]
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)

        selected_hint = ""
        if selected_text and selected_text.strip():
            selected_hint = f"\n用户选中的文本：{selected_text[:200]}"

        prompt = f"""请将以下用户查询改写为一个独立的、适合检索文档的查询语句。
要求：
- 消解代词和指代（如"它"、"这个方法"等），使查询独立可理解
- 保持原始意图不变
- 直接输出改写后的查询，不要加任何前缀或解释
- 如果查询已经足够清晰，直接返回原文

{f'对话历史：{chr(10)}{history_text}' if history_text else ''}
{selected_hint}

当前查询：{rewritten}

改写后的查询："""

        try:
            from services.chat_service import call_ai_api
            from models.provider_registry import PROVIDER_CONFIG

            if not model:
                model = "gpt-4o-mini"
            if not provider:
                provider = "openai"
            if not endpoint:
                endpoint = PROVIDER_CONFIG.get(provider, {}).get("endpoint", "")

            response = await call_ai_api(
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                max_tokens=100,
                temperature=0.3,
            )

            if isinstance(response, dict):
                if response.get("error"):
                    logger.warning(f"[LLM QueryRewrite] 调用失败: {response['error']}")
                    return rewritten
                content = response.get("content", "")
                if not content and "choices" in response:
                    choices = response["choices"]
                    if choices and isinstance(choices, list):
                        content = choices[0].get("message", {}).get("content", "")
            else:
                content = str(response) if response else ""

            content = content.strip()
            if content and len(content) > 3 and content != query:
                logger.info(f"[LLM QueryRewrite] '{query}' → '{content}'")
                return content

            return rewritten

        except Exception as e:
            logger.warning(f"[LLM QueryRewrite] 失败，降级为 regex 改写: {e}")
            return rewritten

    def _extract_key_content(self, text: str, max_chars: int = 50) -> str:
        """从选中文本中提取关键内容片段

        提取策略：
        1. 去除首尾空白
        2. 取第一个有意义的句子或短语
        3. 如果超过 max_chars 则截断并添加省略号

        Args:
            text: 选中的文本
            max_chars: 最大字符数，默认 50

        Returns:
            提取的关键内容片段
        """
        if not text or not text.strip():
            return ''

        # 去除首尾空白和多余换行
        cleaned = text.strip()

        # 按句子分隔符切分，取第一个有意义的句子
        # 支持中英文标点
        sentence_delimiters = re.compile(r'[。！？\n；;!?]')
        sentences = sentence_delimiters.split(cleaned)

        # 找到第一个非空句子
        first_sentence = ''
        for s in sentences:
            s = s.strip()
            if s:
                first_sentence = s
                break

        if not first_sentence:
            first_sentence = cleaned

        # 截断到 max_chars
        if len(first_sentence) > max_chars:
            return first_sentence[:max_chars]

        return first_sentence
