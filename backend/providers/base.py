from abc import ABC, abstractmethod
from typing import List, Optional


class BaseProvider(ABC):
    """统一的Provider接口"""

    @abstractmethod
    async def chat(self, messages: List[dict], api_key: str, model: str, timeout: Optional[float] = None, stream: bool = False, max_tokens: int = 8192, temperature: float = 0.7, top_p: float = 1.0) -> dict:
        raise NotImplementedError
