"""Load SWE-bench Pro Public, SWE-bench-Live, and SWT-Bench.

These benchmarks share the same instance schema: each task is a real GitHub
PR with `repo`, `base_commit`, `problem_statement`, `patch` (gold diff),
`test_patch`, and FAIL_TO_PASS / PASS_TO_PASS test selectors.

Execution requires Docker (per-task images). This module only materializes
the data side — each task becomes a directory with:

  tasks_bench/<variant>/<instance_id>/
    prompt.md          # problem statement + repo + base_commit
    instance.json      # full record for scoring
    gold.diff          # reference patch (for sanity / oracle runs)

To score predictions, use the official `swebench` harness:

  python -m swebench.harness.run_evaluation \
      --predictions_path results/preds_<variant>.jsonl \
      --dataset_name ScaleAI/SWE-bench_Pro --split test ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VARIANTS = {
    "swebench_pro": ("ScaleAI/SWE-bench_Pro", "test"),
    "swebench_live_lite": ("SWE-bench-Live/SWE-bench-Live", "lite"),
    "swebench_live_verified": ("SWE-bench-Live/SWE-bench-Live", "verified"),
    "swebench_live_full": ("SWE-bench-Live/SWE-bench-Live", "full"),
    "swtbench_verified": ("princeton-nlp/SWE-bench_Verified", "test"),  # SWT-Bench reuses SWE-bench instances
    "swtbench_lite": ("princeton-nlp/SWE-bench_Lite", "test"),
}


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _stringify(v):
    """Coerce dataset cells into JSON-safe scalars/strings."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return json.dumps(v, default=str)


def materialize(variant: str, out_root: Path, limit: int | None = None) -> int:
    from datasets import load_dataset

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}; options: {sorted(VARIANTS)}")
    ds_name, split = VARIANTS[variant]
    ds = load_dataset(ds_name, split=split)
    rows = list(ds)
    if limit is not None:
        rows = rows[:limit]

    out_root.mkdir(parents=True, exist_ok=True)
    index = []
    n = 0
    for r in rows:
        inst_id = r.get("instance_id") or r.get("id")
        if not inst_id:
            continue
        task_id = _safe_id(inst_id)
        td = out_root / task_id
        td.mkdir(parents=True, exist_ok=True)

        # Plain-string copy of the full record so callers can reload it.
        record = {k: _stringify(r[k]) for k in r}
        (td / "instance.json").write_text(json.dumps(record, indent=2) + "\n")

        problem = r.get("problem_statement") or ""
        repo = r.get("repo") or ""
        base = r.get("base_commit") or ""
        f2p = r.get("FAIL_TO_PASS") or r.get("fail_to_pass") or []
        p2p = r.get("PASS_TO_PASS") or r.get("pass_to_pass") or []
        prompt_md = (
            f"# {task_id}\n\n"
            f"Repo: `{repo}`\n"
            f"Base commit: `{base}`\n\n"
            "## Problem statement\n\n"
            f"{problem}\n\n"
            "## Tests required to pass\n"
            f"{json.dumps(f2p, default=str, indent=2)}\n\n"
            "## Tests that must still pass\n"
            f"{json.dumps(p2p, default=str, indent=2)}\n\n"
            "## Output\n"
            "Produce a unified diff in `prediction.diff` that, when applied to "
            "the repo at the base commit, makes the FAIL_TO_PASS tests pass "
            "and keeps the PASS_TO_PASS tests passing.\n"
        )
        (td / "prompt.md").write_text(prompt_md)

        gold = r.get("patch")
        if gold:
            (td / "gold.diff").write_text(gold if gold.endswith("\n") else gold + "\n")

        index.append(
            {
                "instance_id": inst_id,
                "task_id": task_id,
                "repo": repo,
                "base_commit": base,
                "dockerhub_tag": r.get("dockerhub_tag"),
            }
        )
        n += 1

    (out_root / "instances.jsonl").write_text(
        "\n".join(json.dumps(x) for x in index) + "\n"
    )
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS.keys()))
    p.add_argument("--out", default="tasks_bench")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    out_root = Path(args.out) / args.variant
    n = materialize(args.variant, out_root, limit=args.limit)
    print(f"wrote {n} instances to {out_root}")
    print(f"  index: {out_root / 'instances.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
