from abc import ABC, abstractmethod
from typing import List, Optional


class BaseProvider(ABC):
    """统一的Provider接口"""

    @abstractmethod
    async def chat(self, messages: List[dict], api_key: str, model: str, timeout: Optional[float] = None, stream: bool = False) -> dict:
        raise NotImplementedError
