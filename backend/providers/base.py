from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseProvider(ABC):
    """统一的Provider接口"""

    @abstractmethod
    async def chat(
        self,
        messages: List[dict],
        api_key: str,
        model: str,
        timeout: Optional[float] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        custom_params: Optional[Dict] = None,
        reasoning_effort: Optional[str] = None,
    ) -> dict:
        """调用 AI 模型进行对话

        Args:
            messages: 对话消息列表
            api_key: API 密钥
            model: 模型名称
            timeout: 请求超时时间（秒）
            stream: 是否流式输出
            max_tokens: 最大输出 token 数，None 表示使用模型默认值
            temperature: 温度参数，None 表示使用模型默认值
            top_p: 核采样参数，None 表示使用模型默认值
            custom_params: 自定义参数字典，直接合并到请求体
            reasoning_effort: 深度思考力度（'low'|'medium'|'high'），None 表示不启用
        """
        raise NotImplementedError
