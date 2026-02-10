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
