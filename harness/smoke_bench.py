"""Sanity check: drop each benchmark's canonical solution into the materialized
task and confirm public + hidden tests pass. Run after `load_benchmark`.

Usage:
  python -m harness.smoke_bench --name humaneval
  python -m harness.smoke_bench --name mbpp
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .load_benchmark import fetch_humaneval, fetch_mbpp, fetch_mbpp_sanitized
from .sandbox import Sandbox, score_hidden


def _evalplus_he_iter():
    from evalplus.data import get_human_eval_plus
    return list(get_human_eval_plus().values())


def _evalplus_mbpp_iter():
    from evalplus.data import get_mbpp_plus
    return list(get_mbpp_plus().values())


def _evalplus_canonical(p):
    return p["prompt"] + p["canonical_solution"]


def _evalplus_he_id(p):
    return p["task_id"].replace("/", "_")


def _evalplus_mbpp_id(p):
    return p["task_id"].replace("/", "_")


def _humaneval_canonical(p: dict) -> str:
    # canonical_solution is the body that completes p["prompt"].
    return p["prompt"] + p["canonical_solution"]


def _mbpp_canonical(p: dict) -> str:
    return p["code"]


FETCHERS = {
    "humaneval": (fetch_humaneval, _humaneval_canonical, lambda p: p["task_id"].replace("/", "_")),
    "mbpp": (fetch_mbpp, _mbpp_canonical, lambda p: f"MBPP_{p['task_id']}"),
    "mbpp_san": (fetch_mbpp_sanitized, _mbpp_canonical, lambda p: f"MBPP_{p['task_id']}"),
    "humaneval_plus": (_evalplus_he_iter, _evalplus_canonical, _evalplus_he_id),
    "mbpp_plus": (_evalplus_mbpp_iter, _evalplus_canonical, _evalplus_mbpp_id),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(FETCHERS.keys()))
    ap.add_argument("--root", default="tasks_bench")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=60, help="pytest timeout per task")
    ap.add_argument("--verbose", action="store_true", help="print each materialized task as it is checked")
    args = ap.parse_args()

    fetch, canonical, id_fn = FETCHERS[args.name]
    problems = fetch()
    if isinstance(problems, dict):
        problems = list(problems.values())
    if args.limit is not None:
        problems = problems[: args.limit]

    root = Path(args.root) / args.name
    n_ok = n_bad = 0
    failures = []
    for i, p in enumerate(problems, start=1):
        task_id = id_fn(p)
        task_dir = root / task_id
        if not task_dir.exists():
            continue
        if args.verbose:
            print(f"[{i}/{len(problems)}] {task_id} ... ", end="", flush=True)
        sb = Sandbox(task_dir)
        try:
            sb.write("solution.py", canonical(p))
            pub = sb.run_pytest("public_tests.py", timeout=args.timeout)
            hid = score_hidden(task_dir, sb, timeout=args.timeout)
            if pub.returncode == 0 and hid.returncode == 0:
                n_ok += 1
                if args.verbose:
                    print("ok")
            else:
                n_bad += 1
                failures.append((task_id, pub.returncode, hid.returncode, hid.stdout[-400:]))
                if args.verbose:
                    print(f"FAIL pub={pub.returncode} hid={hid.returncode}")
        finally:
            sb.cleanup()
    print(f"canonical pass: {n_ok}/{n_ok + n_bad}")
    for tid, pc, hc, tail in failures[:5]:
        print(f"  FAIL {tid} pub={pc} hid={hc}")
        print(re.sub(r"^", "    ", tail, flags=re.MULTILINE))
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
