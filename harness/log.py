"""JSONL run logger."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunRecord:
    task_id: str
    model: str
    mode: str
    k: int
    passed_public: Optional[bool] = None
    passed_self: Optional[bool] = None
    passed_hidden: Optional[bool] = None
    iterations_used: int = 0
    bash_calls_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_time_s: float = 0.0
    final_error_type: Optional[str] = None
    seed: int = 0
    extra: dict = field(default_factory=dict)


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: RunRecord) -> None:
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(record)) + "\n")


def now() -> float:
    return time.monotonic()
