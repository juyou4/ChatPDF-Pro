"""
关键词提取器 - 从用户查询中提取关键词并维护频率统计

设计说明：
- 复用 bm25_service._tokenize 进行中英文混合分词
- 过滤停用词和过短的 token（长度 < 2）
- 维护关键词频率统计，自动识别用户关注领域
- 零 LLM 调用，纯规则+统计方法
"""
from datetime import datetime, timezone
from typing import List

from services.bm25_service import _tokenize


# 中英文停用词表（常见无意义词）
STOP_WORDS = {
    # 中文停用词
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "什么", "那", "这个", "那个", "怎么", "如何", "为什么", "哪些",
    "可以", "能", "吗", "呢", "吧", "啊", "嗯", "哦", "请", "帮",
    "告诉", "解释", "介绍", "描述", "分析", "总结", "关于",
    # 英文停用词
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
    "in", "with", "to", "for", "of", "not", "no", "can", "had", "has",
    "have", "it", "its", "was", "were", "be", "been", "being", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "this", "that", "these", "those", "what", "how", "why", "where",
    "when", "who", "whom", "if", "then", "than", "so", "as", "by",
    "from", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "each", "all", "both", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "too",
    "very", "just", "because", "while", "here", "there", "are", "am",
}


class KeywordExtractor:
    """从用户查询中提取关键词并维护频率统计"""

    def extract_keywords(self, query: str) -> List[str]:
        """从查询文本中提取有意义的关键词

        复用 bm25_service._tokenize 进行分词，
        过滤停用词和过短的 token（长度 < 2）

        Args:
            query: 用户查询文本

        Returns:
            去重后的关键词列表
        """
        if not query or not query.strip():
            return []

        tokens = _tokenize(query)

        # 过滤停用词和过短 token
        keywords = []
        seen = set()
        for token in tokens:
            if len(token) < 2:
                continue
            if token in STOP_WORDS:
                continue
            if token not in seen:
                seen.add(token)
                keywords.append(token)

        return keywords

    def update_frequency(self, profile: dict, keywords: List[str]) -> dict:
        """更新用户画像中的关键词频率统计

        Args:
            profile: 用户画像字典，包含 keyword_frequencies 字段
            keywords: 待更新的关键词列表

        Returns:
            更新后的 profile 字典
        """
        if "keyword_frequencies" not in profile:
            profile["keyword_frequencies"] = {}

        freq = profile["keyword_frequencies"]
        for kw in keywords:
            freq[kw] = freq.get(kw, 0) + 1

        profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        return profile

    def get_focus_areas(self, profile: dict, threshold: int = 3) -> List[str]:
        """获取超过频率阈值的关键词作为用户关注领域

        Args:
            profile: 用户画像字典，包含 keyword_frequencies 字段
            threshold: 频率阈值，默认 3

        Returns:
            超过阈值的关键词列表，按频率降序排列
        """
        freq = profile.get("keyword_frequencies", {})
        focus = [(kw, count) for kw, count in freq.items() if count >= threshold]
        # 按频率降序排列
        focus.sort(key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in focus]
