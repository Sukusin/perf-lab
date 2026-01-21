from abc import ABC, abstractmethod
from typing import Any, Dict

class Backend(ABC):
    @abstractmethod
    def load(self, model_id: str, dtype: str) -> None: ...

    @abstractmethod
    def build_inputs(self, massages: list[dict]) -> Dict[str, Any]: ...

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], max_new_tokens: int) -> Any: ...

    @abstractmethod
    def info(self) -> Dict[str, Any]: ...