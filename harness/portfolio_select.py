"""Visible-signal portfolio selector for completed benchmark runs.

This post-processes existing C and D_val artifacts into a new P_select row per
task. The selector must not inspect hidden outcomes. It chooses a solution using
public/self/model-error signals, then either rescores that selected solution or
reuses the already-recorded hidden result for fast diagnostics.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Optional

from .log import JsonlLogger, RunRecord, now
from .sandbox import Sandbox, score_hidden


_NATURAL_PART_RE = __import__("re").compile(r"(\d+)")


def _natural_task_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    parts = _NATURAL_PART_RE.split(path.name)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in parts
        if part
    )


def _load_rows(path: Path, batch_tag: str) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        raise SystemExit(f"log not found: {path}")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("extra", {}).get("batch_tag") == batch_tag:
            rows.append(row)
    return rows


def _completed_select_keys(path: Path, batch_tag: str) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("mode") != "P_select":
            continue
        if row.get("extra", {}).get("batch_tag") != batch_tag:
            continue
        completed.add((str(row.get("task_id")), int(row.get("seed", 0))))
    return completed


def _artifact_solution_path(row: dict) -> Optional[Path]:
    artifact_dir = row.get("extra", {}).get("artifact_dir")
    if not artifact_dir:
        return None
    root = Path(artifact_dir)
    iters = int(row.get("iterations_used", 0) or 0)
    candidates: list[Path] = []
    if iters > 0:
        candidates.append(root / f"iter_{iters:02d}" / "solution.py")
    candidates.extend(sorted(root.glob("iter_*/solution.py"), reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _model_error_role(error_type: str | None) -> str | None:
    if not error_type or not error_type.startswith("model_error:"):
        return None
    parts = error_type.split(":", 3)
    return parts[1] if len(parts) > 1 else "model"


def _has_artifact(row: dict | None) -> bool:
    return row is not None and _artifact_solution_path(row) is not None


def _public_ok(row: dict | None) -> bool:
    return bool(row is not None and row.get("passed_public") is True)


def _self_ok(row: dict | None) -> bool:
    return bool(row is not None and row.get("passed_self") is True)


def _fatal_generation_error(row: dict | None) -> bool:
    role = _model_error_role(None if row is None else row.get("final_error_type"))
    return role in {"code", "test", "validator", "model"}


def _repair_error(row: dict | None) -> bool:
    if row is None:
        return False
    extra = row.get("extra", {})
    if extra.get("repair_aborted_after_model_error"):
        return True
    return _model_error_role(row.get("final_error_type")) == "repair"


def _dval_visible_good(row: dict | None) -> bool:
    if row is None:
        return False
    return (
        _public_ok(row)
        and _self_ok(row)
        and row.get("extra", {}).get("self_tests_validated") is not False
        and not _fatal_generation_error(row)
        and not _repair_error(row)
        and _has_artifact(row)
    )


def _c_visible_good(row: dict | None) -> bool:
    return _public_ok(row) and not _fatal_generation_error(row) and _has_artifact(row)


def choose_visible_guard(base_row: dict | None, candidate_row: dict | None) -> tuple[str, str]:
    """Choose without hidden labels.

    Conservative rule:
    - Trust D_val when public+self pass and there was no visible model/repair error.
    - Otherwise fall back to C when C passed public tests.
    - If both are visibly weak, choose the candidate with an artifact and fewer
      visible failures, preferring D_val when it at least passed self-tests.
    """
    if _dval_visible_good(candidate_row):
        return "candidate", "dval_public_self_validated"
    if _c_visible_good(base_row):
        return "base", "dval_visible_failed_or_repair_error_c_public_passed"
    if _has_artifact(candidate_row) and _public_ok(candidate_row):
        return "candidate", "c_not_public_passed_dval_public_passed"
    if _has_artifact(candidate_row) and _self_ok(candidate_row):
        return "candidate", "c_not_public_passed_dval_self_passed"
    if _has_artifact(base_row):
        return "base", "fallback_base_artifact_available"
    if _has_artifact(candidate_row):
        return "candidate", "fallback_candidate_artifact_available"
    return "none", "no_artifacts_available"


def _final_error_from_hidden(
    hidden_passed: bool,
    hidden_timed_out: bool,
    selected_visible_passed: bool | None,
) -> str | None:
    if hidden_passed:
        return None
    if hidden_timed_out:
        return "overfit_hidden_timeout" if selected_visible_passed else "hidden_timeout"
    return "overfit_selected_visible" if selected_visible_passed else "incorrect"


def _copy_selected_artifacts(
    artifact_root: Path,
    batch_tag: str,
    seed: int,
    task_id: str,
    solution_path: Path,
) -> str:
    run_dir = artifact_root / batch_tag / f"seed_{seed}" / task_id / "P_select_k3" / "iter_01"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(solution_path, run_dir / "solution.py")
    return str((run_dir.parent).resolve())


def build_selected_record(
    *,
    task_dir: Path,
    base_row: dict | None,
    candidate_row: dict | None,
    selected_kind: str,
    reason: str,
    out_batch_tag: str,
    score_policy: str,
    hidden_timeout: int,
    save_artifacts: bool,
    artifact_root: Path,
) -> RunRecord:
    source_row = candidate_row if selected_kind == "candidate" else base_row
    if source_row is None:
        source_row = base_row or candidate_row
    if source_row is None:
        raise ValueError(f"no source rows for {task_dir.name}")

    seed = int(source_row.get("seed", 0))
    record = RunRecord(
        task_id=task_dir.name,
        model=str(source_row.get("model")),
        mode="P_select",
        k=int(candidate_row.get("k", base_row.get("k", 3)) if candidate_row else base_row.get("k", 3)),
        seed=seed,
    )
    record.extra["batch_tag"] = out_batch_tag
    record.extra["selection_used_hidden"] = False
    record.extra["selection_strategy"] = "visible_guard"
    record.extra["selection_reason"] = reason
    record.extra["selected_source"] = selected_kind
    record.extra["selected_mode"] = source_row.get("mode")
    record.extra["selected_k"] = source_row.get("k")
    record.extra["base_mode"] = None if base_row is None else base_row.get("mode")
    record.extra["base_k"] = None if base_row is None else base_row.get("k")
    record.extra["candidate_mode"] = None if candidate_row is None else candidate_row.get("mode")
    record.extra["candidate_k"] = None if candidate_row is None else candidate_row.get("k")
    record.extra["base_public_passed"] = None if base_row is None else base_row.get("passed_public")
    record.extra["candidate_public_passed"] = None if candidate_row is None else candidate_row.get("passed_public")
    record.extra["candidate_self_passed"] = None if candidate_row is None else candidate_row.get("passed_self")
    record.extra["candidate_self_tests_validated"] = (
        None if candidate_row is None else candidate_row.get("extra", {}).get("self_tests_validated")
    )
    record.extra["candidate_repair_error"] = _repair_error(candidate_row)
    record.extra["score_policy"] = score_policy

    solution_path = _artifact_solution_path(source_row)
    if solution_path is None:
        record.final_error_type = "selector_missing_artifact"
        record.passed_hidden = False
        return record

    record.passed_public = source_row.get("passed_public")
    record.passed_self = source_row.get("passed_self")
    selected_visible_passed = (
        _dval_visible_good(source_row)
        if source_row.get("mode") == "D_val"
        else _c_visible_good(source_row)
    )
    record.extra["selected_visible_passed"] = selected_visible_passed

    t0 = now()
    if score_policy == "reuse":
        record.passed_hidden = source_row.get("passed_hidden")
        record.extra["hidden_timed_out"] = source_row.get("extra", {}).get("hidden_timed_out")
        record.extra["hidden_timeout_s"] = source_row.get("extra", {}).get("hidden_timeout_s")
        record.final_error_type = _final_error_from_hidden(
            bool(record.passed_hidden),
            bool(record.extra.get("hidden_timed_out")),
            selected_visible_passed,
        )
    else:
        sandbox = Sandbox(task_dir)
        try:
            sandbox.write("solution.py", solution_path.read_text())
            hidden = score_hidden(task_dir, sandbox, timeout=hidden_timeout)
            record.bash_calls_used = sandbox.bash_calls
            record.extra["hidden_timed_out"] = hidden.timed_out
            record.extra["hidden_timeout_s"] = hidden_timeout
            record.passed_hidden = hidden.returncode == 0
            record.final_error_type = _final_error_from_hidden(
                bool(record.passed_hidden),
                hidden.timed_out,
                selected_visible_passed,
            )
        finally:
            sandbox.cleanup()

    record.wall_time_s = round(now() - t0, 3)
    if save_artifacts:
        record.extra["artifact_dir"] = _copy_selected_artifacts(
            artifact_root,
            out_batch_tag,
            seed,
            task_dir.name,
            solution_path,
        )
    return record


def _collect_tasks(bench_root: Path, benchmark: str, limit: int | None) -> list[Path]:
    root = bench_root / benchmark
    if not root.exists():
        raise SystemExit(f"benchmark dir not found: {root}")
    tasks = sorted((p for p in root.iterdir() if p.is_dir()), key=_natural_task_key)
    return tasks[:limit] if limit is not None else tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--batch-tag", required=True)
    parser.add_argument("--out-log", required=True, type=Path)
    parser.add_argument("--out-batch-tag", required=True)
    parser.add_argument("--benchmark", default="humaneval_plus")
    parser.add_argument("--bench-root", default="tasks_bench", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-mode", default="C")
    parser.add_argument("--base-k", type=int, default=3)
    parser.add_argument("--candidate-mode", default="D_val")
    parser.add_argument("--candidate-k", type=int, default=3)
    parser.add_argument("--hidden-timeout", type=int, default=360)
    parser.add_argument("--score-policy", choices=["rescore", "reuse"], default="rescore")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=Path("results/artifacts"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = _load_rows(args.log, args.batch_tag)
    by_key = {
        (row["task_id"], row.get("mode"), int(row.get("k", 0)), int(row.get("seed", 0))): row
        for row in rows
    }
    seeds = sorted({int(row.get("seed", 0)) for row in rows}) or [0]
    tasks = _collect_tasks(args.bench_root, args.benchmark, args.limit)
    completed = _completed_select_keys(args.out_log, args.out_batch_tag) if args.resume else set()
    logger = JsonlLogger(args.out_log)

    written = skipped = missing = 0
    selected_counts: dict[str, int] = {"base": 0, "candidate": 0, "none": 0}
    hidden_results: list[bool] = []
    for seed in seeds:
        for task_dir in tasks:
            if (task_dir.name, seed) in completed:
                skipped += 1
                continue
            base_row = by_key.get((task_dir.name, args.base_mode, args.base_k, seed))
            candidate_row = by_key.get((task_dir.name, args.candidate_mode, args.candidate_k, seed))
            if base_row is None and candidate_row is None:
                missing += 1
                continue
            selected_kind, reason = choose_visible_guard(base_row, candidate_row)
            selected_counts[selected_kind] = selected_counts.get(selected_kind, 0) + 1
            record = build_selected_record(
                task_dir=task_dir,
                base_row=base_row,
                candidate_row=candidate_row,
                selected_kind=selected_kind,
                reason=reason,
                out_batch_tag=args.out_batch_tag,
                score_policy=args.score_policy,
                hidden_timeout=args.hidden_timeout,
                save_artifacts=args.save_artifacts,
                artifact_root=args.artifact_root,
            )
            logger.write(record)
            written += 1
            if record.passed_hidden is not None:
                hidden_results.append(bool(record.passed_hidden))
            status = "PASS" if record.passed_hidden else "FAIL"
            print(
                f"{task_dir.name} seed={seed} selected={record.extra.get('selected_mode')} "
                f"reason={reason} hidden={status}"
            )

    if hidden_results:
        print(
            f"\nwrote {written} P_select rows to {args.out_log}; "
            f"hidden_pass={sum(hidden_results)}/{len(hidden_results)} "
            f"({mean(hidden_results) * 100:.2f}%)"
        )
    else:
        print(f"\nwrote {written} P_select rows to {args.out_log}")
    if skipped:
        print(f"skipped {skipped} completed rows")
    if missing:
        print(f"missing source rows for {missing} task/seed pairs")
    print("selected counts: " + ", ".join(f"{k}={v}" for k, v in sorted(selected_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
