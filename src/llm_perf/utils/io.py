from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    raw_jsonl: Path


def make_run_paths(
    runs_root: Path | str = "results/raw",
    date: str | None = None,
    width: int = 4,
    start: int = 1,
    max_tries: int = 100_000,
) -> RunPaths:
    runs_root = Path(runs_root)
    date_str = date or datetime.now().strftime("%d%m%y")  # ddmmyy
    run_dir = runs_root / date_str
    run_dir.mkdir(parents=True, exist_ok=True)

    for n in range(start, start + max_tries):
        raw_jsonl = run_dir / f"{n:0{width}d}.jsonl"
        try:
            # атомарно создаём пустой файл; если уже есть — исключение
            with raw_jsonl.open("x", encoding="utf-8"):
                pass
            run_id = f"{date_str}/{n:0{width}d}"
            return RunPaths(run_id=run_id, run_dir=run_dir, raw_jsonl=raw_jsonl)
        except FileExistsError:
            continue

    raise RuntimeError(f"Could not allocate run file in {run_dir} after {max_tries} attempts")
