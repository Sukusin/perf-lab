from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Backend

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class HFBackend(Backend):
    def __init__(self, local_files_only: bool = True):
        self.local_files_only = local_files_only
        self.tokenizer = None
        self.model = None
        self.model_id = None
        self.dtype = None
        self.device = None

    def load(self, model_id: str, dtype: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Use GPU only")

        self.model_id = model_id
        self.dtype = dtype
        self.device = "cuda"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=self.local_files_only)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=self.local_files_only,
                torch_dtype=_DTYPE_MAP.get(dtype, torch.float16),
            )
            .to(self.device)
            .eval()
        )

    def build_inputs(self, massages: list[dict]) -> dict[str, Any]:
        assert self.tokenizer is not None and self.model is not None
        inputs = self.tokenizer.apply_chat_template(
            massages,
            add_generation_pronmpt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @torch.inference_mode()
    def generate(self, inputs: dict[str, Any], max_new_tokens: int) -> Any:
        assert self.model is not None
        return self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
        )

    def info(self) -> dict[str, Any]:
        assert self.model is not None
        return {
            "backend": "hf",
            "model": self.model_id,
            "dtype": str(next(self.model.parameters()).dtype),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        }
