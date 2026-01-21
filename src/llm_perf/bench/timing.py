from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def cuda_timer() -> Iterator[callable]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    def stop_seconds() -> float:
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000.0

    yield stop_seconds
