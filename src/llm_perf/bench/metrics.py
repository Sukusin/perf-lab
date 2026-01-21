from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Sample:
    dt: float
    new_tokens: int
    tok_s: float
    peak_memory_gb: float


def peak_mem_gb() -> float:
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def compute_sample(dt: float, prompt_len: int, total_len: int) -> Sample:
    new_tokens = int(total_len - prompt_len)
    tok_s = float(new_tokens / dt) if dt > 0 else float("inf")
    return Sample(
        dt=float(dt),
        new_tokens=new_tokens,
        tok_s=tok_s,
        peak_memory_gb=peak_mem_gb(),
    )


def percentile(values, q: float) -> float:
    return float(np.percentile(list(values), q))
