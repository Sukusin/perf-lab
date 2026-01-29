from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .base import Backend

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}


@dataclass
class VLLMConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    vllm_dtype: str = "auto"
    max_model_len: int = 4096
    trust_remote_code: bool = False  # by default
    enforce_eager: bool = False  # by default
    seed: int = 0


class VLLMBackend(Backend):
    def __init__(self, cfg: VLLMConfig | None = None) -> None:
        self.cfg = cfg or VLLMConfig()
        self.model_id: str | None = None
        self.dtype: str | None = None

        self.llm = None
        self.tokenizer = None

    def load(self, model_id: str, dtype: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Use GPU only")
        self.model_id = model_id
        self.dtype = dtype

        vllm_dtype = _DTYPE_MAP.get(dtype, "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
        )
        self.llm = LLM(
            model=model_id,
            dtype=vllm_dtype,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            max_model_len=self.cfg.max_model_len,
            trust_remote_code=self.cfg.trust_remote_code,
            enforce_eager=self.cfg.enforce_eagerm,
            seed=self.cfg.seed,
        )

    def build_inputs(self, messages: list[dict]) -> dict[str, Any]:
        assert self.tokenizer is not None
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return {"prompt": prompt, "message": messages}

    def generate(self, inputs: dict[str, Any], max_new_tokens: int) -> Any:
        assert self.llm is not None
        prompt = inputs["prompt"]
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
        )
        outputs = self.llm.generate(prompts=[prompt], sampling_params=sampling_params)
        out0 = outputs[0]
        comp0 = out0.outputs[0]

        text = comp0.text

        finish_reason = getattr(comp0, "finish_reason", None)
        token_ids = getattr(comp0, "token_ids", None)
        num_generated_tokens = len(token_ids) if token_ids is not None else None

        return {
            "prompt": prompt,
            "text": text,
            "finish_reason": finish_reason,
            "num_generated_tokens": num_generated_tokens,
            "raw": out0,
        }

    def info(self) -> dict[str, Any]:
        return {
            "backend": "vllm",
            "model": self.model_id,
            "dtype": self.dtype,
            "cfg": self.cfg,
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        }
