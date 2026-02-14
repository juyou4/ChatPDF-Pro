"""
多轮 Agent 式检索服务

参考 paper-burner-x 的 streaming-multi-hop 架构，实现：
- LLM 作为"检索规划助手"，不回答问题，只规划检索策略
- 多轮迭代：每轮执行搜索→评估结果→决定是否需要更多信息
- 丰富的工具集：vector_search, grep, keyword_search, regex_search, boolean_search, fetch, map
- 搜索历史去重，避免重复查询
- 任务追踪 (taskStatus)
- 流式进度反馈
"""

import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from services.retrieval_tools import DocContext, execute_tool

logger = logging.getLogger(__name__)

# Agent 系统提示词模板
_AGENT_SYSTEM_PROMPT = """你是检索规划助手，专门负责规划如何从文档中检索相关内容。

**重要：你的角色定位**
- ⚠️ **你不负责回答用户问题**，你只负责规划如何检索文档内容
- ⚠️ **不要生成最终答案、思维导图或 mermaid 图表**
- ✓ 你的任务：分析用户问题 → 选择合适的检索工具 → 输出 JSON 格式的检索计划
- ✓ 检索到的内容会交给另一个 AI 来回答用户问题

## 工具定义（JSON 格式）

### 搜索工具
- {{"tool":"vector_search","args":{{"query":"语义描述","limit":10}}}}
  用途：智能语义搜索（理解同义词、相关概念）

- {{"tool":"grep","args":{{"query":"具体短语","limit":20,"context":2000,"caseInsensitive":true}}}}
  用途：字面文本搜索（精确关键词匹配）
  支持 OR 逻辑：query 可用 | 分隔，如 "方程|公式|equation"

- {{"tool":"keyword_search","args":{{"keywords":["词1","词2"],"limit":8}}}}
  用途：多关键词加权搜索（BM25 算法）

- {{"tool":"regex_search","args":{{"pattern":"\\\\d{{4}}年","limit":10,"context":1500}}}}
  用途：正则表达式搜索（匹配特定格式如日期、编号）

- {{"tool":"boolean_search","args":{{"query":"(CNN OR RNN) AND 对比 NOT 图像","limit":10,"context":1500}}}}
  用途：布尔逻辑搜索（AND/OR/NOT 组合）

{group_tools}

## 决策原则
- 复杂问题优先多工具并用（同一轮 operations 数组中并发执行）
- vector_search 擅长语义理解，grep 擅长精确匹配，两者结合效果最佳
- **严格检查【搜索历史】避免重复搜索**
- 获取到足够内容后立即 final=true
- 每轮最多 5 个操作，搜索 limit 不超过 20

## 返回格式（严格遵守 JSON）
{{
  "operations": [...],
  "final": true/false,
  "taskStatus": {{
    "completed": ["已完成任务..."],
    "current": "当前任务",
    "pending": ["待做任务..."]
  }}
}}

- operations 为空且 final=true 表示检索完成
- final=false 表示还需要继续检索"""

_GROUP_TOOLS_TEMPLATE = """
### 意群工具
- {{"tool":"fetch","args":{{"groupId":"group-1"}}}}
  用途：获取指定意群详细内容（完整论述、公式、数据）

- {{"tool":"map","args":{{"limit":50}}}}
  用途：获取文档整体结构（意群地图：ID、字数、关键词、摘要）"""


class RetrievalAgent:
    """多轮检索规划 Agent"""

    def __init__(
        self,
        api_key: str,
        model: str,
        provider: str,
        endpoint: str = "",
        max_rounds: int = 5,
        temperature: float = 0.3,
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.endpoint = endpoint
        self.max_rounds = max_rounds
        self.temperature = temperature

    async def run(
        self,
        question: str,
        doc_ctx: DocContext,
        doc_name: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """执行多轮检索，流式 yield 进度和最终上下文

        Yields:
            进度事件: {"type": "retrieval_progress", "phase": str, "message": str}
            最终结果: {"type": "retrieval_complete", "context": str, "detail": list}
        """
        has_groups = bool(doc_ctx.semantic_groups)
        group_tools = _GROUP_TOOLS_TEMPLATE if has_groups else ""

        system_prompt = _AGENT_SYSTEM_PROMPT.format(group_tools=group_tools)

        # 状态
        fetched_content: Dict[str, dict] = {}  # group_id -> {granularity, text}
        search_results: List[str] = []  # 累积的搜索结果片段
        search_history: List[dict] = []  # 搜索历史
        task_status = {"completed": [], "current": "", "pending": []}

        yield {
            "type": "retrieval_progress",
            "phase": "start",
            "message": "正在分析问题，规划检索策略...",
        }

        for round_idx in range(self.max_rounds):
            yield {
                "type": "retrieval_progress",
                "phase": "round_start",
                "round": round_idx + 1,
                "message": f"第 {round_idx + 1} 轮取材...",
            }

            # 构建用户消息
            user_content = self._build_user_message(
                question, doc_name, search_results, search_history,
                fetched_content, task_status, round_idx,
            )

            # 调用 LLM 规划
            yield {
                "type": "retrieval_progress",
                "phase": "planning",
                "round": round_idx + 1,
                "message": "LLM 规划中...",
            }

            plan = await self._call_planner(system_prompt, user_content)
            if plan is None:
                logger.warning(f"[RetrievalAgent] 第 {round_idx + 1} 轮规划失败")
                break

            operations = plan.get("operations", [])
            is_final = plan.get("final", False)

            # 更新任务追踪
            new_status = plan.get("taskStatus")
            if new_status and isinstance(new_status, dict):
                task_status = {
                    "completed": new_status.get("completed", task_status["completed"]),
                    "current": new_status.get("current", ""),
                    "pending": new_status.get("pending", []),
                }

            # 如果没有操作且标记为最终，结束
            if not operations and is_final:
                break

            # 如果没有操作但不是最终，也结束（防止死循环）
            if not operations:
                break

            # 执行操作
            for op in operations[:5]:  # 每轮最多 5 个操作
                tool_name = op.get("tool", "")
                tool_args = op.get("args", {})

                if not tool_name:
                    continue

                # 检查搜索历史去重
                if tool_name in ("vector_search", "grep", "keyword_search"):
                    query_key = tool_args.get("query", "") or str(tool_args.get("keywords", ""))
                    if self._is_duplicate_search(search_history, tool_name, query_key):
                        logger.info(f"[RetrievalAgent] 跳过重复搜索: {tool_name} {query_key}")
                        continue

                yield {
                    "type": "retrieval_progress",
                    "phase": "executing",
                    "round": round_idx + 1,
                    "message": f"执行 {tool_name}...",
                    "tool": tool_name,
                }

                # 执行工具
                result = execute_tool(tool_name, tool_args, doc_ctx)

                # 记录搜索历史
                query_key = tool_args.get("query", "") or tool_args.get("pattern", "") or str(tool_args.get("keywords", "")) or tool_args.get("groupId", "")
                search_history.append({
                    "tool": tool_name,
                    "query": query_key,
                    "resultCount": result.get("result_count", len(result.get("results", []))),
                })

                # 累积结果
                tool_results = result.get("results", [])
                if tool_name == "fetch":
                    group_id = tool_args.get("groupId", "")
                    gran = result.get("granularity", "full")
                    if tool_results:
                        fetched_content[group_id] = {
                            "granularity": gran,
                            "text": tool_results[0],
                        }
                elif tool_name == "map":
                    # map 结果单独存储
                    if tool_results:
                        search_results.append(f"【文档地图】\n{tool_results[0][:3000]}")
                else:
                    for chunk in tool_results:
                        if chunk and chunk not in search_results:
                            search_results.append(chunk)

                yield {
                    "type": "retrieval_progress",
                    "phase": "tool_result",
                    "round": round_idx + 1,
                    "message": result.get("summary", f"{tool_name} 完成"),
                    "tool": tool_name,
                    "result_count": result.get("result_count", 0),
                }

            # 如果标记为最终，结束
            if is_final:
                break

        # 构建最终上下文
        final_context, detail = self._build_final_context(
            search_results, fetched_content
        )

        yield {
            "type": "retrieval_progress",
            "phase": "complete",
            "message": f"检索完成，共获取 {len(search_results)} 个片段，{len(fetched_content)} 个意群",
        }

        yield {
            "type": "retrieval_complete",
            "context": final_context,
            "detail": detail,
            "search_history": search_history,
            "task_status": task_status,
        }

    def _build_user_message(
        self,
        question: str,
        doc_name: str,
        search_results: List[str],
        search_history: List[dict],
        fetched_content: Dict[str, dict],
        task_status: dict,
        round_idx: int,
    ) -> str:
        """构建每轮发送给 planner LLM 的用户消息"""
        parts = []

        parts.append(f"文档名称: {doc_name}")
        parts.append(f"\n用户问题:\n{question}")

        # 搜索历史
        if search_history:
            recent = search_history[-8:]
            history_lines = []
            for s in recent:
                tool_label = {
                    "vector_search": "向量",
                    "keyword_search": "BM25",
                    "grep": "GREP",
                    "regex_search": "正则",
                    "boolean_search": "布尔",
                    "fetch": "获取意群",
                    "map": "文档地图",
                }.get(s["tool"], s["tool"])
                status = f"✓ {s['resultCount']}个结果" if s["resultCount"] > 0 else "✗ 无结果"
                history_lines.append(f"- {tool_label} \"{s['query']}\" → {status}")
            parts.append(f"\n【搜索历史】(避免重复搜索):\n" + "\n".join(history_lines))

        # 任务追踪
        if round_idx > 0 and (task_status["completed"] or task_status["current"]):
            status_parts = []
            if task_status["completed"]:
                status_parts.append(f"已完成: {'; '.join(task_status['completed'])}")
            if task_status["current"]:
                status_parts.append(f"上轮任务: {task_status['current']}")
            if task_status["pending"]:
                status_parts.append(f"待完成: {'; '.join(task_status['pending'])}")
            parts.append(f"\n【任务追踪】\n" + "\n".join(status_parts))

        # 已获取内容摘要
        fetched_summary = "无"
        all_content = []

        # search_results 中的片段
        for i, chunk in enumerate(search_results[:15]):
            preview = chunk[:500] + "..." if len(chunk) > 500 else chunk
            all_content.append(f"[片段{i+1}]\n{preview}")

        # fetched_content 中的意群
        for gid, data in fetched_content.items():
            preview = data["text"][:500] + "..." if len(data["text"]) > 500 else data["text"]
            all_content.append(f"【{gid}】({data['granularity']})\n{preview}")

        if all_content:
            fetched_summary = "\n\n".join(all_content)

        parts.append(f"\n【已获取内容】:\n{fetched_summary}")

        return "\n".join(parts)

    async def _call_planner(self, system_prompt: str, user_content: str) -> Optional[dict]:
        """调用 LLM 获取检索计划"""
        from models.provider_registry import PROVIDER_CONFIG

        endpoint = self.endpoint
        if not endpoint:
            provider_cfg = PROVIDER_CONFIG.get(self.provider, {})
            endpoint = provider_cfg.get("endpoint", "https://api.openai.com/v1/chat/completions")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(endpoint, headers=headers, json=body)
                if response.status_code != 200:
                    logger.error(f"[RetrievalAgent] LLM 调用失败: {response.status_code}")
                    return None
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[RetrievalAgent] LLM 调用异常: {e}")
            return None

        # 解析 JSON（从内容中提取）
        return self._parse_plan_json(content)

    def _parse_plan_json(self, content: str) -> Optional[dict]:
        """从 LLM 输出中解析 JSON 检索计划"""
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试找到第一个 { 和最后一个 }
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(content[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"[RetrievalAgent] 无法解析 JSON: {content[:200]}")
        return None

    def _is_duplicate_search(
        self, history: List[dict], tool_name: str, query_key: str
    ) -> bool:
        """检查是否为重复搜索"""
        if not query_key:
            return False
        for h in history:
            if h["tool"] == tool_name and h["query"] == query_key:
                return True
        return False

    def _build_final_context(
        self,
        search_results: List[str],
        fetched_content: Dict[str, dict],
    ) -> tuple:
        """构建最终上下文和详情

        Returns:
            (context_string, detail_list)
        """
        context_parts = []
        detail = []

        # 添加搜索结果片段（去重，限制总量）
        seen = set()
        for chunk in search_results:
            chunk_key = chunk[:100]
            if chunk_key in seen:
                continue
            seen.add(chunk_key)
            context_parts.append(chunk)

        # 添加意群内容
        for gid, data in fetched_content.items():
            kw_text = ""
            context_parts.append(f"【{gid} - {data['granularity']}】\n{data['text']}")
            detail.append({
                "group_id": gid,
                "granularity": data["granularity"],
                "char_count": len(data["text"]),
            })

        # 限制总长度（约 50000 字符）
        total = 0
        trimmed_parts = []
        for part in context_parts:
            if total + len(part) > 50000:
                remaining = 50000 - total
                if remaining > 500:
                    trimmed_parts.append(part[:remaining] + "...(截断)")
                break
            trimmed_parts.append(part)
            total += len(part)

        context_string = "\n\n".join(trimmed_parts)
        return context_string, detail
