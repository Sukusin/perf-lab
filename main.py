import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import json, time, uuid
from pathlib import Path

RUN_ID = str(uuid.uuid4())
OUT = Path("bench_runs.jsonl")

def log_row(row: dict):
    row = dict(row)
    row["ts"] = time.time()
    row["run_id"] = RUN_ID
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if torch.cuda.is_available():
    device = "cuda"
else:
    print("Use GPU instead of CPU")
    quit()
    


def pctl(xs, q):
    return float(np.percentile(np.array(xs, dtype=np.float64), q)) if xs else float("nan")


@torch.inference_mode()
def run_once(model, inputs, tokenizer, n_new_tokens: int):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    outputs = model.generate(
        **inputs,
        min_new_tokens=n_new_tokens,
        max_new_tokens=n_new_tokens,   # фиксируем длину
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    end.record()
    torch.cuda.synchronize()

    dt_ms = start.elapsed_time(end)
    dt = dt_ms / 1000.0

    prompt_len = inputs["input_ids"].shape[-1]
    total_len = outputs[0].shape[-1]
    new_tokens = int(total_len - prompt_len)

    tok_s = new_tokens / max(dt, 1e-9)
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return dt, tok_s, new_tokens, peak_mem_gb


MODEL = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL,
                                          use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
).eval()


WARMUP_ITERS = 2
RUNS = 5
N_NEW_TOKENS = [128, 256, 2048]

prompt_list = [
    "Describe a forgotten object at the bottom of an old drawer, using sensory details (sight, touch, smell) to hint at its history without explicitly stating it.",
    "Write a micro-story that begins with the sentence: 'The map was wrong, and that was the best thing that could have happened.' Include a specific setting, a clear problem, and a resolution.",
    "Write a short story from the alternating perspectives of two characters who witness the same mysterious event—a sudden, silent power outage across their entire city—but interpret it in radically different ways (one sees science, the other sees magic). Explore their backgrounds, their reactions during the event, and the aftermath where their theories collide.",
]


for j, prompt in enumerate(prompt_list, start=1):
    n_new = N_NEW_TOKENS[j-1]  # вместо s

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    for _ in range(WARMUP_ITERS):
        _ = run_once(model, inputs, tokenizer, n_new_tokens=32)

    times, tok_s_list, mem_list = [], [], []

    for i in range(RUNS):
        dt, tok_s, new_tokens, peak_gb = run_once(model, inputs, tokenizer, n_new_tokens=n_new)
        times.append(dt)
        tok_s_list.append(tok_s)
        mem_list.append(peak_gb)
        print(f"[prompt {j} | {i+1}/{RUNS}] time={dt:.3f}s new_tokens={new_tokens} tok/s={tok_s:.2f} peak_vram={peak_gb:.2f}GB")
        
        log_row({
        "model": MODEL,
        "dtype": "fp16",
        "quant": None,
        "device": torch.cuda.get_device_name(0),
        "prompt_id": j,
        "prompt_len": int(inputs["input_ids"].shape[-1]),
        "new_tokens_target": int(n_new),
        "dt_s": float(dt),
        "new_tokens": int(new_tokens),
        "tok_s": float(tok_s),
        "peak_vram_gb": float(peak_gb),
        })


    print(f"\n=== SUMMARY (prompt {j}) ===")
    print(f"runs: {RUNS}, fixed_new_tokens: {n_new}")
    print(f"time   p50: {pctl(times, 50):.3f}s   p95: {pctl(times, 95):.3f}s")
    print(f"tok/s  p50: {pctl(tok_s_list, 50):.2f}   p05: {pctl(tok_s_list, 5):.2f}   min: {min(tok_s_list):.2f}")
    print(f"peak_vram p50: {pctl(mem_list, 50):.2f}GB  max: {max(mem_list):.2f}GB\n")

