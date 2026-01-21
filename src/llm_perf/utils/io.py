import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    raw_jsonl: Path


def make_run_paths(results_root: Path = Path("results/raw")) -> RunPaths:
    run_id = str(uuid.uuid4())
    run_dir = Path(results_root / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(run_id=run_id, run_dir=run_dir, raw_jsonl=run_dir / "per_run.jsonl")
