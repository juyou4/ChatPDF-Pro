"""
检索工具 JSON Schema 定义模块。

导出 TOOL_SCHEMAS：7 条 OpenAI 标准 function 格式的工具描述，
供 Planner_LLM 原生函数调用（Native Tool Calls）使用。
"""

# OpenAI 标准 function 格式的工具 Schema 列表
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "向量语义搜索，按相关度返回 chunks",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询文本"},
                    "limit": {
                        "type": "integer",
                        "default": 14,
                        "minimum": 1,
                        "maximum": 30,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "BM25 关键词搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "limit": {"type": "integer", "default": 8},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "精确文本子串搜索，query 可用 | 分隔多个关键词表示 OR",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    "context": {"type": "integer", "default": 2000},
                    "caseInsensitive": {"type": "boolean", "default": True},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regex_search",
            "description": "正则表达式搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "context": {"type": "integer", "default": 1500},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "boolean_search",
            "description": "布尔逻辑搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "按 groupId 获取语义组内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "groupId": {"type": "string"},
                    "granularity": {
                        "type": "string",
                        "enum": ["full", "digest", "summary"],
                        "default": "digest",
                    },
                },
                "required": ["groupId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map",
            "description": "返回文档语义组地图，可包含结构线索",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 50},
                    "includeStructure": {"type": "boolean", "default": True},
                },
            },
        },
    },
]
