
# llm-perf-lab

Небольшой и воспроизводимый бенчмарк для LLM-инференса на GPU (пока: HuggingFace Transformers backend).
Считает **TTFT**, **TPOT**, **tok/s**, **p50/p95** и пишет результаты в `results/raw/<run_id>/per_run.jsonl`.

## Что умеет
- Запуск экспериментов из YAML-конфига (`configs/runs/*.yaml`)
- Справочник моделей по алиасам (`configs/models.yaml`)
- Метрики:
  - **TTFT** (time to first token) — приближённо через генерацию 1 токена
  - **TPOT** (time per output token) — по (full_time - ttft) / (N-1)
  - **tok/s**, **peak VRAM**
- Логирование каждого кейса в JSONL + вывод summary в консоль

> Важно: TTFT/TPOT в этой версии измеряются “двухпрогонно” (1 токен и N токенов). Это простой и стабильный старт для сравнения оптимизаций.

## Структура проекта (коротко)
```
.
├── README.md
├── configs
│   ├── models.yaml
│   ├── runs
│   │   └── baseline.yaml
│   └── sweep
│       └── latency_grid.yaml
├── py
├── pyproject.toml
├── results
│   ├── plots
│   ├── raw
│   └── reports
├── src
│   ├── llm_perf
│   │   ├── __init__.py
│   │   ├── analysis
│   │   │   ├── agregate.py
│   │   │   └── plots.py
│   │   ├── backends
│   │   │   ├── base.py
│   │   │   ├── hf.py
│   │   │   ├── trtllm.py
│   │   │   └── vllm.py
│   │   ├── bench
│   │   │   ├── metrics.py
│   │   │   ├── reporters.py
│   │   │   ├── runner.py
│   │   │   └── timing.py
│   │   ├── cli.py
│   │   ├── loadgen
│   │   │   ├── async-client.py
│   │   │   └── traces.py
│   │   ├── profiling
│   │   │   ├── nsys.py
│   │   │   └── torch-profiler.py
│   │   └── utils
│   │       ├── env.py
│   │       ├── io.py
│   │       └── schema.py
└── uv.lock

````

---

## Установка (uv)
Из корня репозитория:

```bash
uv venv
uv pip install -e .
````

Проверка, что CLI доступен:

```bash
llm-perf --help
```

---

## Конфиги

### 1) `configs/models.yaml`

Хранит алиасы моделей и их параметры:

```yaml
models:
  qwen3_4b:
    hf_id: "Qwen/Qwen3-4B"
    local_files_only: true
    dtype_default: "fp16"
```

* `hf_id` — id модели для `transformers`
* `local_files_only: true` — грузим только из локального кеша HF
* `dtype_default` — `fp16` или `bf16`

### 2) `configs/runs/baseline.yaml`

Описывает конкретный прогон:

```yaml
model: "qwen3_4b"
backend: "hf"
warmup: 2
repeats: 5
cases:
  - prompt: "Who are you?"
    new_tokens: 32
```

* `warmup` — прогрев (не логируется)
* `repeats` — число измерений для каждого кейса
* `cases` — список кейсов (prompt + new_tokens)

---

## Запуск

```bash
llm-perf run -c configs/runs/baseline.yaml
```

После запуска:

* в консоли будет вывод по кейсам + summary p50/p95
* появится файл результатов:

```
results/raw/<run_id>/per_run.jsonl
```

---

## Формат результатов (`per_run.jsonl`)

Каждая строка — один измеренный кейс (один repeat). Пример полей:

* `run_id`
* `model`, `backend`, `dtype`
* `prompt_len`
* `new_tokens`
* `ttft_s`, `tpot_s`, `dt_full_s`, `tok_s`
* `peak_memory_allocated_gb`
* `gpu`, `cuda_version`, версии библиотек (env snapshot)

---

## Частые проблемы

### `ModuleNotFoundError: No module named 'llm_perf'`

Проверь, что проект установлен editable:

```bash
uv pip install -e . --reinstall
```

### CUDA недоступна

Этот бенчмарк рассчитан на GPU. Если CUDA не видна, проверь:

* `nvidia-smi`
* что в окружении стоит CUDA-сборка PyTorch
* что запускаешь не CPU-only окружение

---

## Куда расширять дальше

* Добавить backend: `src/llm_perf/backends/vllm.py` / `trtllm.py`
* Добавить sweep-конфиги и генерацию сетки экспериментов
* Добавить агрегацию результатов (`analysis/aggregate.py`) и графики (`analysis/plots.py`)
