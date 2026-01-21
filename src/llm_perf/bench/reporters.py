import json
from pathlib import Path
from typing import Any, Dict

class JsonlReporter:
    def __init__(self, out_path: Path, run_id: str):
        self.out_path = out_path
        self.run_id = run_id

    def log(self, row:Dict[str, any]) -> None:
        row = dict(row)
        row["run_id"] = self.run_id
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")