"""
高级搜索服务 - 正则表达式搜索和布尔逻辑搜索

参考 paper-burner-x 的 AdvancedSearchTools 实现，
提供正则表达式搜索和布尔逻辑搜索（AND/OR/NOT）功能。

功能：
- regex_search: 正则表达式全文匹配搜索，返回匹配结果和上下文片段
- boolean_search: 布尔逻辑搜索（AND/OR/NOT），按相关性分数降序排列
"""

import re
from typing import List


class AdvancedSearchService:
    """高级搜索服务，支持正则表达式搜索和布尔逻辑搜索"""

    def regex_search(
        self,
        pattern: str,
        text: str,
        limit: int = 20,
        context_chars: int = 200,
    ) -> List[dict]:
        """正则表达式搜索，返回匹配结果和上下文片段

        参数:
            pattern: 正则表达式模式
            text: 要搜索的文本
            limit: 最大返回结果数，默认 20
            context_chars: 上下文片段的前后字符数，默认 200

        返回:
            匹配结果列表，每项包含:
            - match_text: 匹配的文本
            - match_offset: 匹配在全文中的偏移量
            - context_snippet: 上下文片段
            - score: 相关性分数（正则搜索固定为 1.0）

        异常:
            ValueError: 正则表达式语法无效时抛出
        """
        if not pattern or not text:
            return []

        # 编译正则表达式，语法无效时抛出 ValueError
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            raise ValueError(f"正则表达式语法错误: {e}")

        results = []
        count = 0

        for match in regex.finditer(text):
            if count >= limit:
                break

            match_text = match.group(0)
            match_start = match.start()
            match_end = match.end()

            # 跳过零宽度匹配（避免无限循环）
            if match_start == match_end:
                continue

            # 提取上下文片段
            context_start = max(0, match_start - context_chars)
            context_end = min(len(text), match_end + context_chars)
            context_snippet = text[context_start:context_end]

            results.append({
                "match_text": match_text,
                "match_offset": match_start,
                "context_snippet": context_snippet,
                "score": 1.0,
            })

            count += 1

        return results

    def boolean_search(
        self,
        query: str,
        text: str,
        limit: int = 20,
        context_chars: int = 200,
    ) -> List[dict]:
        """布尔逻辑搜索（AND/OR/NOT），返回匹配结果和上下文片段

        支持的布尔操作符：
        - AND: 两个词项都必须出现（在窗口范围内）
        - OR: 任一词项出现即可（加分项）
        - NOT: 排除包含该词项的结果

        参数:
            query: 布尔查询表达式，如 "CNN AND 对比 NOT 图像"
            text: 要搜索的文本
            limit: 最大返回结果数，默认 20
            context_chars: 上下文片段的前后字符数，默认 200

        返回:
            匹配结果列表（按 score 降序排列），每项包含:
            - match_text: 匹配的文本
            - match_offset: 匹配在全文中的偏移量
            - context_snippet: 上下文片段
            - score: 相关性分数
        """
        if not query or not text:
            return []

        # 解析布尔查询，提取 must/should/not 词项
        must_terms, should_terms, not_terms = self._parse_boolean_query(query)

        # 如果没有任何有效词项，返回空结果
        if not must_terms and not should_terms:
            return []

        # 查找所有词项在文本中的位置
        text_lower = text.lower()
        term_positions = {}
        all_terms = set(must_terms + should_terms + not_terms)
        for term in all_terms:
            term_positions[term] = self._find_term_positions(
                term.lower(), text_lower
            )

        # 确定基础搜索词项（must 优先，否则用 should 的第一个）
        if must_terms:
            base_terms = must_terms
        else:
            base_terms = [should_terms[0]]
            should_terms = should_terms[1:]

        # 以第一个基础词项的位置为锚点，评估布尔条件
        base_positions = term_positions.get(base_terms[0], [])
        window_size = 500  # 布尔条件的窗口范围（字符数）

        results = []
        seen_positions = set()

        for base_pos in base_positions:
            base_start = base_pos["start"]
            base_end = base_pos["end"]

            # 检查所有 must 词项是否在窗口范围内
            is_valid = True
            matched_terms = [base_terms[0]]
            min_pos = base_start
            max_pos = base_end
            score = 1.0

            for term in base_terms[1:]:
                positions = term_positions.get(term, [])
                nearby = self._find_nearby_position(
                    positions, base_start, window_size
                )
                if nearby is None:
                    is_valid = False
                    break
                matched_terms.append(term)
                min_pos = min(min_pos, nearby["start"])
                max_pos = max(max_pos, nearby["end"])
                score += 1.0

            if not is_valid:
                continue

            # 检查 should 词项（加分项）
            for term in should_terms:
                positions = term_positions.get(term, [])
                nearby = self._find_nearby_position(
                    positions, base_start, window_size
                )
                if nearby is not None:
                    matched_terms.append(term)
                    min_pos = min(min_pos, nearby["start"])
                    max_pos = max(max_pos, nearby["end"])
                    score += 0.5

            # 检查 NOT 词项（排除）
            for term in not_terms:
                positions = term_positions.get(term, [])
                nearby = self._find_nearby_position(
                    positions, base_start, window_size
                )
                if nearby is not None:
                    is_valid = False
                    break

            if not is_valid:
                continue

            # 去重：避免同一位置重复出现
            pos_key = (min_pos, max_pos)
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)

            # 提取匹配文本和上下文片段
            match_text = text[min_pos:max_pos]
            context_start = max(0, min_pos - context_chars)
            context_end = min(len(text), max_pos + context_chars)
            context_snippet = text[context_start:context_end]

            results.append({
                "match_text": match_text,
                "match_offset": min_pos,
                "context_snippet": context_snippet,
                "score": score,
            })

        # 按 score 降序排列
        results.sort(key=lambda x: x["score"], reverse=True)

        # 限制结果数量
        return results[:limit]

    def _parse_boolean_query(self, query: str):
        """解析布尔查询表达式，提取 must/should/not 词项

        参数:
            query: 布尔查询字符串，如 "CNN AND 对比 NOT 图像"

        返回:
            (must_terms, should_terms, not_terms) 三元组
        """
        # 标准化空白字符
        query = query.strip()

        # 分词：按 AND/OR/NOT 操作符分割
        tokens = self._tokenize_boolean_query(query)

        must_terms = []
        should_terms = []
        not_terms = []
        current_operator = "AND"  # 默认操作符

        for token in tokens:
            if token["type"] == "operator":
                current_operator = token["value"]
            elif token["type"] == "term":
                term_value = token["value"].strip()
                if not term_value:
                    continue
                if current_operator == "NOT":
                    not_terms.append(term_value)
                    current_operator = "AND"  # 重置为默认
                elif current_operator == "OR":
                    should_terms.append(term_value)
                    current_operator = "AND"  # 重置为默认
                else:  # AND 或默认
                    must_terms.append(term_value)

        return must_terms, should_terms, not_terms

    def _tokenize_boolean_query(self, query: str) -> List[dict]:
        """将布尔查询字符串分词为 token 列表

        支持的 token 类型：
        - term: 搜索词项
        - operator: AND/OR/NOT 操作符

        参数:
            query: 布尔查询字符串

        返回:
            token 列表，每项包含 type 和 value
        """
        tokens = []
        # 使用正则匹配 AND/OR/NOT 操作符（前后需要空白或边界）
        # 操作符必须大写
        parts = re.split(r'\b(AND|OR|NOT)\b', query)

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part in ("AND", "OR", "NOT"):
                tokens.append({"type": "operator", "value": part})
            else:
                tokens.append({"type": "term", "value": part})

        return tokens

    def _find_term_positions(
        self, term_lower: str, text_lower: str
    ) -> List[dict]:
        """查找词项在文本中的所有出现位置（大小写不敏感）

        参数:
            term_lower: 小写化的搜索词项
            text_lower: 小写化的文本

        返回:
            位置列表，每项包含 start 和 end
        """
        positions = []
        pos = 0
        while True:
            idx = text_lower.find(term_lower, pos)
            if idx == -1:
                break
            positions.append({
                "start": idx,
                "end": idx + len(term_lower),
            })
            pos = idx + 1
        return positions

    def _find_nearby_position(
        self,
        positions: List[dict],
        anchor: int,
        window_size: int,
    ) -> dict:
        """在位置列表中查找距离锚点最近且在窗口范围内的位置

        参数:
            positions: 位置列表
            anchor: 锚点位置
            window_size: 窗口大小（字符数）

        返回:
            最近的位置字典，或 None（无匹配）
        """
        best = None
        best_distance = float("inf")
        for pos in positions:
            distance = abs(pos["start"] - anchor)
            if distance <= window_size and distance < best_distance:
                best = pos
                best_distance = distance
        return best
