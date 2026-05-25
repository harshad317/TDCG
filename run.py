#!/usr/bin/env python3
"""CLI entry point.

Usage:
  python run.py --model qwen2.5-coder:1.5b --tasks tasks/task_001_sum_evens --modes A,C --ks 1,3,5
  python run.py --model qwen2.5-coder:1.5b --all --modes A,C --ks 1,3,5

Logs JSONL to results/runs.jsonl by default.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from harness.agent import AgentConfig, run_task
from harness.analysis import build_ablation_summary, format_ablation_summary
from harness.log import JsonlLogger, RunRecord
from harness.models import build_client
from harness.plot import default_batch_tag, filter_rows, load, make_all


@dataclass(frozen=True)
class WorkItem:
    index: int
    seed: int
    task_dir: Path
    mode: str
    k: int


_NATURAL_PART_RE = re.compile(r"(\d+)")
REFERENCE_REQUIRED_BENCHMARKS = {
    "humaneval",
    "humaneval_plus",
    "mbpp",
    "mbpp_plus",
    "mbpp_san",
}


def natural_task_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    """Sort benchmark task ids in human order: HumanEval_2 before HumanEval_10."""
    parts = _NATURAL_PART_RE.split(path.name)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in parts
        if part
    )


def collect_tasks(args) -> list[Path]:
    if args.benchmark:
        root = Path(args.bench_root) / args.benchmark
        if not root.exists():
            raise SystemExit(
                f"benchmark dir {root} not found — run "
                f"`python -m harness.load_benchmark --name {args.benchmark}` first"
            )
        return sorted((p for p in root.iterdir() if p.is_dir()), key=natural_task_key)
    if args.all:
        tasks_root = Path("tasks")
        return sorted((p for p in tasks_root.iterdir() if p.is_dir()), key=natural_task_key)
    return [Path(t) for t in args.tasks]


def assert_reference_tasks_available(args, tasks: list[Path], modes: list[str]) -> None:
    if args.benchmark not in REFERENCE_REQUIRED_BENCHMARKS or "D_val" not in modes:
        return
    missing = [p.name for p in tasks if not (p / "reference_solution.py").exists()]
    if not missing:
        return
    shown = ", ".join(missing[:10])
    if len(missing) > 10:
        shown += f", ... (+{len(missing) - 10} more)"
    limit_arg = f" --limit {args.limit}" if args.limit is not None else ""
    raise SystemExit(
        "D_val needs reference_solution.py for generated HumanEval/MBPP tasks. "
        "Some selected task directories are stale or incomplete: "
        f"{shown}. Regenerate the benchmark with "
        f"`python -m harness.load_benchmark --name {args.benchmark}"
        f"{limit_arg} --out {args.bench_root}`."
    )


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
    if record.mode == "D_val" and record.extra.get("self_tests_validated") is False:
        return "INVALID"
    if record.mode in ("D", "D_sep", "D_dual", "D_val", "E") and record.extra.get("self_tests_present") is False:
        return "MISSING"
    if record.mode in ("D", "D_sep", "D_dual", "D_val", "E") and record.extra.get("ran_self_tests") is False:
        return "NOT_RUN"
    return fmt_status(record.passed_self)


def fmt_hidden_status(record: RunRecord) -> str:
    if record.extra.get("hidden_timed_out") is True:
        return "TIMEOUT"
    return fmt_status(record.passed_hidden)


def fmt_repair_status(record: RunRecord) -> str:
    if record.extra.get("score_hidden_each_iter") is not True:
        return ""
    suffix = ""
    if record.extra.get("repair_aborted_after_model_error"):
        suffix = " repair_abort=model_error"
        if record.extra.get("repair_model_timeout_count", 0):
            suffix = " repair_abort=timeout"
    if record.final_error_type == "validator_invalid_self_tests":
        return " repair=SKIPPED" + suffix
    if record.extra.get("fixed_by_self_tests") is True:
        return " repair=FIXED" + suffix
    if record.extra.get("hidden_improved") is True:
        return " repair=IMPROVED" + suffix
    if record.extra.get("initial_hidden_pass") is True and record.passed_hidden is True:
        return " repair=ALREADY_PASS" + suffix
    if record.extra.get("initial_hidden_pass") is False and record.passed_hidden is False:
        return " repair=NOT_FIXED" + suffix
    return " repair=UNKNOWN" + suffix


def format_record_result(record: RunRecord) -> str:
    hp = record.passed_hidden
    badge = "PASS" if hp else ("FAIL" if hp is False else "??")
    return (
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


def format_work_prefix(item: WorkItem) -> str:
    return (
        f"[{item.index}] seed={item.seed} task={item.task_dir.name} "
        f"mode={item.mode} k={item.k} ... "
    )


def build_work_items(tasks: list[Path], modes: list[str], ks: list[int], seeds: list[int]) -> list[WorkItem]:
    items: list[WorkItem] = []
    total = 0
    for seed in seeds:
        for task_dir in tasks:
            for mode in modes:
                mode_ks = [1] if mode in ("A", "B") else ks
                for k in mode_ks:
                    total += 1
                    items.append(WorkItem(total, seed, task_dir, mode, k))
    return items


def work_key(item: WorkItem) -> tuple[str, str, int, int]:
    return (item.task_dir.name, item.mode, item.k, item.seed)


def load_completed_work_keys(
    path: Path,
    *,
    batch_tag: str,
    model: str,
) -> tuple[set[tuple[str, str, int, int]], int]:
    completed: set[tuple[str, str, int, int]] = set()
    invalid_rows = 0
    if not path.exists():
        return completed, invalid_rows

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if row.get("model") != model:
                continue
            extra = row.get("extra") or {}
            if extra.get("batch_tag") != batch_tag:
                continue
            try:
                completed.add((
                    str(row["task_id"]),
                    str(row["mode"]),
                    int(row["k"]),
                    int(row.get("seed", 0)),
                ))
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
    return completed, invalid_rows


def make_record(
    args,
    item: WorkItem,
    batch_tag: str,
    validator_model: str | None,
    repair_model: str | None,
    repair_model_timeout: int | None,
) -> RunRecord:
    record = RunRecord(
        task_id=item.task_dir.name,
        model=args.model,
        mode=item.mode,
        k=item.k,
        seed=item.seed,
    )
    record.extra["batch_tag"] = batch_tag
    record.extra["model_timeout_s"] = args.model_timeout
    if args.cheap_dval:
        record.extra["cheap_dval"] = True
    if item.mode in ("D_dual", "D_val"):
        record.extra["test_model"] = args.test_model
    if item.mode == "D_val":
        record.extra["validator_model"] = validator_model
    if item.mode in ("D_dual", "D_val") and (repair_model or repair_model_timeout is not None):
        record.extra["repair_model"] = repair_model or args.model
        record.extra["repair_model_timeout_s"] = repair_model_timeout or args.model_timeout
    return record


def make_agent_config(args, item: WorkItem, batch_tag: str) -> AgentConfig:
    return AgentConfig(
        mode=item.mode,
        k=item.k,
        max_bash_calls=args.max_bash_calls,
        pytest_timeout=args.pytest_timeout,
        hidden_timeout=args.hidden_timeout,
        score_hidden_each_iter=args.score_hidden_each_iter,
        code_skill_path=Path(args.code_skill),
        test_skill_path=Path(args.test_skill),
        validator_skill_path=Path(args.validator_skill),
        artifact_root=(
            Path(args.artifact_root) / batch_tag
            if args.save_artifacts
            else None
        ),
        repair_candidates=args.repair_candidates,
        self_test_candidates=args.self_test_candidates,
        code_candidates=args.code_candidates,
        abort_repair_on_model_error=not args.continue_repair_after_model_error,
    )


def execute_work_item(
    args,
    item: WorkItem,
    batch_tag: str,
    validator_model: str | None,
    repair_model: str | None,
    repair_model_timeout: int | None,
) -> RunRecord:
    record = make_record(args, item, batch_tag, validator_model, repair_model, repair_model_timeout)
    if args.dry_run:
        record.extra["dry_run"] = True
        return record

    try:
        client = build_client(
            args.model,
            backend=args.backend,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=item.seed,
            timeout=args.model_timeout,
        )
        test_client = None
        if args.test_model:
            test_client = build_client(
                args.test_model,
                backend=args.backend,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=item.seed,
                timeout=args.model_timeout,
            )
        validator_client = None
        if validator_model:
            validator_client = build_client(
                validator_model,
                backend=args.backend,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=item.seed,
                timeout=args.model_timeout,
            )
        repair_client = None
        if repair_model or repair_model_timeout is not None:
            repair_client = build_client(
                repair_model or args.model,
                backend=args.backend,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=item.seed,
                timeout=repair_model_timeout or args.model_timeout,
            )
        run_task(
            client,
            item.task_dir,
            make_agent_config(args, item, batch_tag),
            record,
            test_client=test_client,
            validator_client=validator_client,
            repair_client=repair_client,
        )
    except Exception as e:
        record.final_error_type = f"harness_error:{type(e).__name__}:{e}"
    return record


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="code model id, e.g. qwen2.5-coder:1.5b")
    p.add_argument("--test-model", default=None,
                   help="test-writing model id; required for modes D_dual and D_val")
    p.add_argument("--validator-model", default=None,
                   help="test-validation model id; defaults to --test-model for mode D_val")
    p.add_argument("--repair-model", default=None,
                   help="optional model id used only for failed D_dual/D_val repairs; defaults to --model")
    p.add_argument("--repair-model-timeout", type=int, default=None,
                   help="timeout in seconds for repair model calls; defaults to --model-timeout")
    p.add_argument("--backend", default="ollama")
    p.add_argument("--tasks", nargs="*", default=[], help="task directories")
    p.add_argument("--all", action="store_true", help="run every task under tasks/")
    p.add_argument("--benchmark", default=None,
                   choices=["humaneval", "mbpp", "mbpp_san", "humaneval_plus",
                            "mbpp_plus", "livecodebench", "bigcodebench_hard", "bigcodebench"],
                   help="run an official benchmark from <bench_root>/<name>/")
    p.add_argument("--bench-root", default="tasks_bench")
    p.add_argument("--limit", type=int, default=None,
                   help="only run the first N tasks (after natural/numeric sorting)")
    p.add_argument("--modes", default="A,C", help="comma-separated modes: A,B,C,D,D_sep,D_dual,D_val,E")
    p.add_argument("--ks", default="1,3,5", help="comma-separated k values; A/B forced to k=1")
    p.add_argument("--max-bash-calls", type=int, default=20)
    p.add_argument("--pytest-timeout", type=int, default=10,
                   help="timeout for visible public/self pytest feedback runs")
    p.add_argument("--hidden-timeout", type=int, default=60,
                   help="timeout for final hidden benchmark scoring")
    p.add_argument("--model-timeout", type=int, default=120,
                   help="timeout in seconds for each model HTTP request")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel worker count for independent task/mode/k cases")
    p.add_argument("--score-hidden-each-iter", action="store_true",
                   help="silently score hidden tests after each iteration for repair-causality analysis; never shown to the model")
    p.add_argument("--repair-candidates", type=int, default=3,
                   help="independent repair candidates to generate and visibly test per failed D_val/D_dual repair step")
    p.add_argument("--self-test-candidates", type=int, default=1,
                   help="independent self-test suites to generate, validate, and merge for D_val")
    p.add_argument("--code-candidates", type=int, default=1,
                   help="independent initial solution candidates to generate and select with visible tests for D_dual/D_val")
    p.add_argument("--continue-repair-after-model-error", action="store_true",
                   help="keep attempting later repairs after a repair model request errors; default is fail-fast for cost control")
    p.add_argument("--cheap-dval", action="store_true",
                   help="cost-capped D_val preset: one self-test suite, one code candidate, one repair candidate, max 12 bash calls, 180s repair timeout")
    p.add_argument("--save-artifacts", action="store_true",
                   help="save per-iteration solution.py, self_tests.py, and pytest outputs for debugging")
    p.add_argument("--artifact-root", default="results/artifacts",
                   help="root directory for --save-artifacts output")
    p.add_argument("--code-skill", default="agent_skills/code_writer/skills.md",
                   help="skill markdown used by the code-writing model in D_dual/D_val")
    p.add_argument("--test-skill", default="agent_skills/test_writer/skills.md",
                   help="skill markdown used by the test-writing model in D_dual/D_val")
    p.add_argument("--validator-skill", default="agent_skills/test_validator/skills.md",
                   help="skill markdown used by the test-validation model in D_val")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0,
                   help="model sampling seed for deterministic backends")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds; overrides --seed and repeats the full run per seed")
    p.add_argument("--log", default="results/runs.jsonl")
    p.add_argument("--resume", action="store_true",
                   help="skip task/mode/k/seed rows already present in --log for this batch tag and model")
    p.add_argument("--resume-log", default=None,
                   help="JSONL file to read completed rows from; implies --resume and defaults to --log")
    p.add_argument("--dry-run", action="store_true", help="skip model calls, just verify wiring")
    p.add_argument("--batch-tag", default=None, help="label for this batch; default = timestamp")
    p.add_argument("--no-plot", action="store_true", help="skip auto-plotting at end")
    p.add_argument("--plot-dir", default=None, help="override plot output dir")
    args = p.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.cheap_dval:
        args.self_test_candidates = 1
        args.code_candidates = 1
        args.repair_candidates = 1
        args.max_bash_calls = min(args.max_bash_calls, 12)
        if args.repair_model_timeout is None:
            args.repair_model_timeout = 180
    if args.repair_model_timeout is not None and args.repair_model_timeout < 1:
        raise SystemExit("--repair-model-timeout must be >= 1")
    if args.repair_candidates < 1:
        raise SystemExit("--repair-candidates must be >= 1")
    if args.self_test_candidates < 1:
        raise SystemExit("--self-test-candidates must be >= 1")
    if args.code_candidates < 1:
        raise SystemExit("--code-candidates must be >= 1")
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
    if any(mode in modes for mode in ("D_dual", "D_val")) and not args.test_model:
        raise SystemExit("--test-model is required when using mode D_dual or D_val")
    assert_reference_tasks_available(args, tasks, modes)
    validator_model = args.validator_model or args.test_model
    repair_model = args.repair_model
    repair_model_timeout = args.repair_model_timeout

    logger = JsonlLogger(Path(args.log))
    work_items = build_work_items(tasks, modes, ks, seeds)
    total_work_items = len(work_items)

    if args.resume or args.resume_log:
        resume_log = Path(args.resume_log or args.log)
        completed, invalid_rows = load_completed_work_keys(
            resume_log,
            batch_tag=batch_tag,
            model=args.model,
        )
        work_items = [item for item in work_items if work_key(item) not in completed]
        skipped = total_work_items - len(work_items)
        print(
            f"resume: skipped {skipped}/{total_work_items} completed cases "
            f"from {resume_log}"
        )
        if invalid_rows:
            print(
                f"resume: ignored {invalid_rows} malformed rows in {resume_log}",
                file=sys.stderr,
            )

    if args.jobs == 1:
        for item in work_items:
            print(format_work_prefix(item), end="", flush=True)
            record = execute_work_item(
                args,
                item,
                batch_tag,
                validator_model,
                repair_model,
                repair_model_timeout,
            )
            logger.write(record)
            print("dry" if args.dry_run else format_record_result(record))
    else:
        print(
            f"running {len(work_items)} cases with {args.jobs} workers "
            f"(model timeout {args.model_timeout}s"
            + (
                f", repair timeout {repair_model_timeout}s"
                if repair_model_timeout is not None
                else ""
            )
            + ")"
        )
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    execute_work_item,
                    args,
                    item,
                    batch_tag,
                    validator_model,
                    repair_model,
                    repair_model_timeout,
                ): item
                for item in work_items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as e:
                    record = make_record(
                        args,
                        item,
                        batch_tag,
                        validator_model,
                        repair_model,
                        repair_model_timeout,
                    )
                    record.final_error_type = f"harness_error:{type(e).__name__}:{e}"
                logger.write(record)
                status = "dry" if args.dry_run else format_record_result(record)
                print(format_work_prefix(item) + status, flush=True)

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
