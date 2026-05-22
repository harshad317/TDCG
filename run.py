#!/usr/bin/env python3
"""CLI entry point.

Usage:
  python run.py --model qwen2.5-coder:1.5b --tasks tasks/task_001_sum_evens --modes A,C --ks 1,3,5
  python run.py --model qwen2.5-coder:1.5b --all --modes A,C --ks 1,3,5

Logs JSONL to results/runs.jsonl by default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness.agent import AgentConfig, run_task
from harness.analysis import build_ablation_summary, format_ablation_summary
from harness.log import JsonlLogger, RunRecord
from harness.models import build_client
from harness.plot import default_batch_tag, filter_rows, load, make_all


def collect_tasks(args) -> list[Path]:
    if args.benchmark:
        root = Path(args.bench_root) / args.benchmark
        if not root.exists():
            raise SystemExit(
                f"benchmark dir {root} not found — run "
                f"`python -m harness.load_benchmark --name {args.benchmark}` first"
            )
        return sorted(p for p in root.iterdir() if p.is_dir())
    if args.all:
        tasks_root = Path("tasks")
        return sorted(p for p in tasks_root.iterdir() if p.is_dir())
    return [Path(t) for t in args.tasks]


def parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def fmt_status(value: bool | None) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "-"


def fmt_self_status(record: RunRecord) -> str:
    if record.mode in ("D", "E") and record.extra.get("self_tests_present") is False:
        return "MISSING"
    if record.mode in ("D", "E") and record.extra.get("ran_self_tests") is False:
        return "NOT_RUN"
    return fmt_status(record.passed_self)


def fmt_hidden_status(record: RunRecord) -> str:
    if record.extra.get("hidden_timed_out") is True:
        return "TIMEOUT"
    return fmt_status(record.passed_hidden)


def fmt_repair_status(record: RunRecord) -> str:
    if record.extra.get("score_hidden_each_iter") is not True:
        return ""
    if record.extra.get("fixed_by_self_tests") is True:
        return " repair=FIXED"
    if record.extra.get("hidden_improved") is True:
        return " repair=IMPROVED"
    if record.extra.get("initial_hidden_pass") is True and record.passed_hidden is True:
        return " repair=ALREADY_PASS"
    if record.extra.get("initial_hidden_pass") is False and record.passed_hidden is False:
        return " repair=NOT_FIXED"
    return " repair=UNKNOWN"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Ollama model id, e.g. qwen2.5-coder:1.5b")
    p.add_argument("--backend", default="ollama")
    p.add_argument("--tasks", nargs="*", default=[], help="task directories")
    p.add_argument("--all", action="store_true", help="run every task under tasks/")
    p.add_argument("--benchmark", default=None,
                   choices=["humaneval", "mbpp", "mbpp_san", "humaneval_plus",
                            "mbpp_plus", "livecodebench", "bigcodebench_hard", "bigcodebench"],
                   help="run an official benchmark from <bench_root>/<name>/")
    p.add_argument("--bench-root", default="tasks_bench")
    p.add_argument("--limit", type=int, default=None,
                   help="only run the first N tasks (after sorting)")
    p.add_argument("--modes", default="A,C", help="comma-separated modes: A,B,C,D,E")
    p.add_argument("--ks", default="1,3,5", help="comma-separated k values; A/B forced to k=1")
    p.add_argument("--max-bash-calls", type=int, default=10)
    p.add_argument("--pytest-timeout", type=int, default=10,
                   help="timeout for visible public/self pytest feedback runs")
    p.add_argument("--hidden-timeout", type=int, default=60,
                   help="timeout for final hidden benchmark scoring")
    p.add_argument("--score-hidden-each-iter", action="store_true",
                   help="silently score hidden tests after each iteration for repair-causality analysis; never shown to the model")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0,
                   help="model sampling seed for deterministic backends")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds; overrides --seed and repeats the full run per seed")
    p.add_argument("--log", default="results/runs.jsonl")
    p.add_argument("--dry-run", action="store_true", help="skip model calls, just verify wiring")
    p.add_argument("--batch-tag", default=None, help="label for this batch; default = timestamp")
    p.add_argument("--no-plot", action="store_true", help="skip auto-plotting at end")
    p.add_argument("--plot-dir", default=None, help="override plot output dir")
    args = p.parse_args()
    batch_tag = args.batch_tag or default_batch_tag()

    tasks = collect_tasks(args)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks specified (use --tasks, --all, or --benchmark)", file=sys.stderr)
        return 2

    modes = parse_csv_str(args.modes)
    ks = parse_csv_ints(args.ks)
    seeds = parse_csv_ints(args.seeds) if args.seeds else [args.seed]

    logger = JsonlLogger(Path(args.log))

    total = 0
    for seed in seeds:
        client = build_client(
            args.model,
            backend=args.backend,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=seed,
        )
        for task_dir in tasks:
            for mode in modes:
                mode_ks = [1] if mode in ("A", "B") else ks
                for k in mode_ks:
                    total += 1
                    record = RunRecord(
                        task_id=task_dir.name,
                        model=args.model,
                        mode=mode,
                        k=k,
                        seed=seed,
                    )
                    record.extra["batch_tag"] = batch_tag
                    print(
                        f"[{total}] seed={seed} task={task_dir.name} mode={mode} k={k} ... ",
                        end="",
                        flush=True,
                    )
                    if args.dry_run:
                        record.extra["dry_run"] = True
                        logger.write(record)
                        print("dry")
                        continue
                    cfg = AgentConfig(
                        mode=mode,
                        k=k,
                        max_bash_calls=args.max_bash_calls,
                        pytest_timeout=args.pytest_timeout,
                        hidden_timeout=args.hidden_timeout,
                        score_hidden_each_iter=args.score_hidden_each_iter,
                    )
                    try:
                        run_task(client, task_dir, cfg, record)
                    except Exception as e:
                        record.final_error_type = f"harness_error:{type(e).__name__}:{e}"
                    logger.write(record)
                    hp = record.passed_hidden
                    badge = "PASS" if hp else ("FAIL" if hp is False else "??")
                    print(
                        f"{badge} "
                        f"public={fmt_status(record.passed_public)} "
                        f"self={fmt_self_status(record)} "
                        f"hidden={fmt_hidden_status(record)} "
                        f"iters={record.iterations_used} "
                        f"bash={record.bash_calls_used} "
                        f"t={record.wall_time_s}s"
                        f"{fmt_repair_status(record)}"
                        + (f" error={record.final_error_type}" if record.final_error_type else "")
                    )

    if not args.dry_run:
        rows = filter_rows(load(Path(args.log)), batch_tag=batch_tag)
        if rows:
            summary = build_ablation_summary(rows)
            print("\n" + format_ablation_summary(summary))

    if not args.no_plot and not args.dry_run:
        plot_dir = Path(args.plot_dir) if args.plot_dir else Path("results/plots") / batch_tag
        plots = make_all(Path(args.log), plot_dir, batch_tag=batch_tag)
        if plots:
            print(f"\nwrote {len(plots)} plots to {plot_dir}")
            print(f"wrote ablation summary to {plot_dir / 'ablation_summary.json'}")
        else:
            print("\nno plots generated (no data matched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
