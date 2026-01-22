from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from ..backends.hf import HFBackend
from ..utils.env import snapshot_env
from ..utils.io import make_run_paths
from .metrics import compute_sample, percentile
from .reporters import JsonlReporter
from .timing import cuda_timer


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty YAML: {path}")
    return data


def run_from_config(run_path: Path | str, models_path: Path | str = Path("configs/models.yaml")) -> None:
    run_path = Path(run_path)
    models_path = Path(models_path)

    run_cfg = _load_yaml(run_path)
    models_cfg = _load_yaml(models_path)

    models = models_cfg.get("models", {})
    model_alias = run_cfg.get("model")
    if model_alias not in models:
        raise KeyError(f"There is no {model_alias} in {models_path}")
    model_spec = models.get(model_alias)
    model_id = model_spec.get("hf_id")

    backend_id = run_cfg.get("backend", "hf")
    if backend_id != "hf":
        raise NotImplementedError('Only "hf" backend realized yet')

    dtype = run_cfg.get("dtype") or model_spec.get("dtype_default")
    local_files_only = run_cfg.get("local_files_only", False)
    if local_files_only is None:
        local_files_only = bool(model_spec.get("local_files_only"))
    else:
        local_files_only = bool(local_files_only)

    warmup_n = int(run_cfg.get("warmup", 2))
    repeats = int(run_cfg.get("repeats", 5))

    cases = run_cfg.get("cases", [])
    if not cases:
        raise ValueError(f"There are no prompts in {run_cfg}")

    paths = make_run_paths()
    reporter = JsonlReporter(paths.raw_jsonl, paths.run_id)

    backend = HFBackend(local_files_only=local_files_only)
    backend.load(model_id=model_id, dtype=dtype)

    env = snapshot_env()
    info = backend.info()

    warmup_inputs = backend.build_inputs([{"role": "user", "content": "warmup"}])
    for _ in range(warmup_n):
        backend.generate(inputs=warmup_inputs, max_new_tokens=16)

    for _id, case in enumerate(tqdm(cases, desc="Cases: ")):
        # print(f"Case {id+1}/{len(cases)}")
        prompt = case.get("prompt")
        new_tokens = int(case.get("new_tokens", 32))

        inputs = backend.build_inputs([{"role": "user", "content": prompt}])
        prompt_len = int(inputs["input_ids"].shape[-1])

        times: list[float] = []
        tok_s_list: list[float] = []

        for _ in tqdm(range(repeats), desc="Repeats: ", leave=False):
            torch.cuda.reset_peak_memory_stats()
            with cuda_timer() as stop:
                outputs = backend.generate(inputs=inputs, max_new_tokens=new_tokens)
                dt = stop()

            total_len = int(outputs[0].shape[-1])
            sample = compute_sample(dt=dt, prompt_len=prompt_len, total_len=total_len)

            times.append(sample.dt)
            tok_s_list.append(sample.tok_s)

            reporter.log(
                {
                    **env,
                    **info,
                    "config_path": str(run_path),
                    "models_path": str(models_path),
                    "model_alias": model_alias,
                    "local_files_only": local_files_only,
                    "prompt": prompt,
                    "prompt_len": prompt_len,
                    "target_new_tokens": new_tokens,
                    "new_tokens": sample.new_tokens,
                    "tok_s": sample.tok_s,
                    "dt": sample.dt,
                    "peak_memory_allocated_gb": sample.peak_memory_gb,
                }
            )

        print("----")
        print(f"prompt_len={prompt_len} target_new_tokens={new_tokens}")
        print(f"times p50={percentile(times, 50):.4f}s  p95={percentile(times, 95):.4f}s")
        print(f"tok/s  p50={percentile(tok_s_list, 50):.2f}  p95={percentile(tok_s_list, 95):.2f}")
        print("----")


if __name__ == "__main__":
    run_from_config(run_path="configs/runs/baseline.yaml")
