from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    @abstractmethod
    def load(self, model_id: str, dtype: str) -> None: ...

    @abstractmethod
    def build_inputs(self, massages: list[dict]) -> dict[str, Any]: ...

    @abstractmethod
    def generate(self, inputs: dict[str, Any], max_new_tokens: int) -> Any: ...

    @abstractmethod
    def info(self) -> dict[str, Any]: ...
