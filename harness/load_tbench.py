"""Load Terminal-Bench (terminal-bench-core).

Each task is a fully self-contained Docker scenario: Dockerfile, task.yaml
(natural-language instruction + setup), tests/, solution.sh (gold), and a
docker-compose.yaml. We do NOT copy the assets — the official harness already
knows how to find them via the `terminal-bench` python package cache.

For each task we materialize a thin pointer dir in tasks_bench/terminal_bench/
containing:

  prompt.md           # the instruction extracted from task.yaml
  tbench_path.txt     # absolute path to the cached Terminal-Bench task dir
  manifest.json       # task name, version, paths

Execution is delegated to the official `tb` CLI / `terminal-bench` runner.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read_instruction(task_dir: Path) -> str:
    yml = task_dir / "task.yaml"
    if not yml.exists():
        return ""
    text = yml.read_text()
    # Prefer pyyaml if available, else fall back to a regex grab.
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        for k in ("instruction", "prompt", "task", "description"):
            if data.get(k):
                return str(data[k])
    except Exception:
        pass
    m = re.search(r"^\s*(instruction|prompt|description)\s*:\s*\|(.*?)(?:^\S|\Z)", text, re.M | re.S)
    if m:
        return m.group(2).strip()
    return text[:2000]


def materialize(out_root: Path, version: str = "0.1.1", limit: int | None = None) -> int:
    from terminal_bench.dataset import Dataset

    ds = Dataset(name="terminal-bench-core", version=version)
    tasks = list(ds.tasks)
    if limit is not None:
        tasks = tasks[:limit]

    out_root.mkdir(parents=True, exist_ok=True)
    index = []
    for tb_path in tasks:
        task_id = tb_path.name
        td = out_root / task_id
        td.mkdir(parents=True, exist_ok=True)
        instruction = _read_instruction(tb_path)
        (td / "prompt.md").write_text(
            f"# {task_id}\n\n"
            f"Terminal-Bench task. Source dir: `{tb_path}`\n\n"
            f"## Instruction\n\n{instruction}\n\n"
            f"## How to evaluate\n\n"
            f"Run the official Terminal-Bench harness (`tb run`) against this "
            f"task. The agent must produce a sequence of shell commands that "
            f"satisfies `tests/test_outputs.py` (or equivalent) in the task dir.\n"
        )
        (td / "tbench_path.txt").write_text(str(tb_path) + "\n")
        (td / "manifest.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "dataset": "terminal-bench-core",
                    "version": version,
                    "source_path": str(tb_path),
                    "files": sorted(p.name for p in tb_path.iterdir()),
                },
                indent=2,
            )
            + "\n"
        )
        index.append({"task_id": task_id, "source_path": str(tb_path)})

    (out_root / "instances.jsonl").write_text(
        "\n".join(json.dumps(x) for x in index) + "\n"
    )
    return len(tasks)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tasks_bench")
    p.add_argument("--version", default="0.1.1")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    out_root = Path(args.out) / "terminal_bench"
    n = materialize(out_root, version=args.version, limit=args.limit)
    print(f"wrote {n} task pointers to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
