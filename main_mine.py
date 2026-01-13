# Load model directly
import torch
import sys
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    print("Use GPU only")
    quit()

def logging(row):
    row = dict(row)
    
    


def run_once(model, inputs, n_new_tokens):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = model.generate(**inputs, max_new_tokens=n_new_tokens, min_new_tokens=n_new_tokens)
    end.record()
    # print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
    torch.cuda.synchronize()
    dt = start.elapsed_time(end)/1000

    prompt_length = inputs["input_ids"].shape[-1]
    all_token = outputs[0].shape[-1]
    new_tokens = all_token - prompt_length
    tok_s = new_tokens / dt
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return dt, tok_s, peak_memory_gb, new_tokens

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",
                                                local_files_only=True,
                                                torch_dtype=torch.float16
                                                ).to(device).eval()

    print(next(model.parameters()).device)

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
        
    for _ in range(2):
        run_once(model, inputs, n_new_tokens=32)

    times, tok_s_list = [], []
    for _ in range(5):
        print("---------------------------------")
        dt, tok_s, peak_memory_gb, new_tokens = run_once(model, inputs, n_new_tokens=32)

        times.append(dt)
        tok_s_list.append(tok_s)

        print(f"New tokens:       {new_tokens}")
        print(f"Tok/s:            {tok_s:.04f} tok/s")
        print(f"Time:             {dt:.04f} s")
        print(f"Memory allocated: {peak_memory_gb:.02f}")

times_p50 = float(np.percentile(times, 50))
tok_s_p50 = float(np.percentile(tok_s_list, 50))
times_p95 = float(np.percentile(times, 95))
tok_s_p95 = float(np.percentile(tok_s_list, 95))
print(f"Calculated times p50: {times_p50:.04f}         tok/s: {tok_s_p50:.04f}")
print(f"Calculated times p95: {times_p95:.04f}         tok/s: {tok_s_p95:.04f}")