import torch

from ..backends.hf import HFBackend
from ..utils.env import snapshot_env
from ..utils.io import make_run_paths
from .metrics import compute_sample, percentile
from .reporters import JsonlReporter
from .timing import cuda_timer


def run():
    paths = make_run_paths()
    reporter = JsonlReporter(paths.raw_jsonl, paths.run_id)

    model_id = "Qwen/Qwen3-4B"
    dtype = "float16"
    warmup_n = 2
    repeats = 5
    new_tokens = 32

    backend = HFBackend(local_files_only=True)
    backend.load(model_id=model_id, dtype=dtype)

    massages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = backend.build_inputs(massages=massages)
    prompt_len = int(inputs["input_ids"].shape[-1])

    for _ in range(warmup_n):
        backend.generate(inputs=inputs, max_new_tokens=16)

    times: list[float] = []
    tok_s_list: list[float] = []

    for s in range(repeats):
        print(f"Run {s+1}/{repeats}")
        torch.cuda.reset_peak_host_memory_stats()
        with cuda_timer() as stop:
            outputs = backend.generate(inputs=inputs, max_new_tokens=new_tokens)
            dt = stop()

        total_len = int(outputs[0].shape[-1])
        sample = compute_sample(dt=dt, prompt_len=prompt_len, total_len=total_len)
        times.append(sample.dt)
        tok_s_list.append(sample.tok_s)

        reporter.log(
            {
                **backend.info(),
                **snapshot_env(),
                "prompt_len": prompt_len,
                "new_tokens": new_tokens,
                "tok_s": sample.tok_s,
                "dt": sample.dt,
                "peak_memory_allocated_gb": sample.peak_memory_gb,
            }
        )

    print(f"times p50={percentile(times, 50):.4f}s  p95={percentile(times, 95):.4f}s")
    print(f"tok/s p50={percentile(tok_s_list, 50):.4f}  p95={percentile(tok_s_list, 95):.4f}")


if __name__ == "__main__":
    run()
