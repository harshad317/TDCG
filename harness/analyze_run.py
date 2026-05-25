"""Summarize a benchmark JSONL run for feedback-harness diagnostics."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _load_rows(path: Path, batch_tag: str | None) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if batch_tag is not None and row.get("extra", {}).get("batch_tag") != batch_tag:
            continue
        rows.append(row)
    return rows


def _label(row: dict) -> str:
    return f"{row['mode']}/k={row['k']}"


def _pct(n: int, d: int) -> str:
    return "-" if d == 0 else f"{100.0 * n / d:.1f}%"


def _hidden_pass(row: dict | None) -> bool | None:
    return None if row is None else row.get("passed_hidden")


def _model_error_role(error_type: str | None) -> str | None:
    if not error_type or not error_type.startswith("model_error:"):
        return None
    parts = error_type.split(":", 3)
    return parts[1] if len(parts) > 1 else "model"


def summarize(rows: list[dict]) -> str:
    by_label: dict[str, list[dict]] = defaultdict(list)
    by_task: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        label = _label(row)
        by_label[label].append(row)
        by_task[(row["task_id"], row.get("seed", 0))][label] = row

    lines = ["# Harness Diagnostic Summary", ""]
    lines.append("| Mode | Cases | Hidden Pass | Avg Iter | Avg Bash | Fixed | Overfit | Incorrect |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label in sorted(by_label):
        group = by_label[label]
        n = len(group)
        hidden_passes = sum(1 for r in group if r.get("passed_hidden") is True)
        avg_iter = mean(r.get("iterations_used", 0) or 0 for r in group)
        avg_bash = mean(r.get("bash_calls_used", 0) or 0 for r in group)
        fixed = sum(1 for r in group if r.get("extra", {}).get("fixed_by_self_tests") is True)
        overfit = sum(1 for r in group if str(r.get("final_error_type", "")).startswith("overfit"))
        incorrect = sum(1 for r in group if r.get("final_error_type") == "incorrect")
        lines.append(
            f"| {label} | {n} | {_pct(hidden_passes, n)} | {avg_iter:.2f} | "
            f"{avg_bash:.2f} | {fixed} | {overfit} | {incorrect} |"
        )

    baseline_label = "A/k=1"
    if any(baseline_label in task_rows for task_rows in by_task.values()):
        lines += ["", "## Paired Against A/k=1", ""]
        lines.append("| Mode | n | Helped | Hurt | Both Pass | Both Fail | Net Lift |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for label in sorted(label for label in by_label if label != baseline_label):
            helped = hurt = both_pass = both_fail = n = 0
            for task_rows in by_task.values():
                base = task_rows.get(baseline_label)
                cur = task_rows.get(label)
                if base is None or cur is None:
                    continue
                base_pass = _hidden_pass(base)
                cur_pass = _hidden_pass(cur)
                if base_pass is None or cur_pass is None:
                    continue
                n += 1
                if not base_pass and cur_pass:
                    helped += 1
                elif base_pass and not cur_pass:
                    hurt += 1
                elif base_pass and cur_pass:
                    both_pass += 1
                else:
                    both_fail += 1
            net = "-" if n == 0 else f"{100.0 * (helped - hurt) / n:+.2f}pp"
            lines.append(
                f"| {label} | {n} | {helped} | {hurt} | {both_pass} | {both_fail} | {net} |"
            )

    lines += ["", "## Self-Test Vs Hidden", ""]
    lines.append("| Mode | Accepted Good | Missed Bug | False Alarm | Caught Bug |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label in sorted(by_label):
        accepted_good = missed_bug = false_alarm = caught_bug = 0
        for row in by_label[label]:
            self_pass = row.get("passed_self")
            hidden_pass = row.get("passed_hidden")
            if self_pass is True and hidden_pass is True:
                accepted_good += 1
            elif self_pass is True and hidden_pass is False:
                missed_bug += 1
            elif self_pass is False and hidden_pass is True:
                false_alarm += 1
            elif self_pass is False and hidden_pass is False:
                caught_bug += 1
        lines.append(f"| {label} | {accepted_good} | {missed_bug} | {false_alarm} | {caught_bug} |")

    candidate_rows = [r for r in rows if r.get("extra", {}).get("repair_candidate_search_count")]
    if candidate_rows:
        tested = sum(r.get("extra", {}).get("repair_candidates_tested_count", 0) for r in candidate_rows)
        selected = sum(
            r.get("extra", {}).get("repair_candidates_selected_visible_pass_count", 0)
            for r in candidate_rows
        )
        lines += [
            "",
            "## Repair Candidate Search",
            "",
            f"Rows with repair search: {len(candidate_rows)}",
            f"Candidates tested: {tested}",
            f"Selected visible-pass candidates: {selected}",
        ]

    model_error_rows = [
        r for r in rows
        if _model_error_role(r.get("final_error_type")) is not None
        or r.get("extra", {}).get("repair_model_error_count", 0)
    ]
    if model_error_rows:
        lines += [
            "",
            "## Model Error Control",
            "",
            "| Mode | Final Code/Test/Validator Errors | Repair Errors | Repair Timeouts | Preserved Current Solution |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label in sorted(by_label):
            group = by_label[label]
            final_model_errors = sum(
                1 for r in group
                if _model_error_role(r.get("final_error_type")) in {"code", "test", "validator", "model"}
            )
            repair_errors = sum(r.get("extra", {}).get("repair_model_error_count", 0) or 0 for r in group)
            repair_timeouts = sum(r.get("extra", {}).get("repair_model_timeout_count", 0) or 0 for r in group)
            preserved = sum(r.get("extra", {}).get("repair_preserved_solution_count", 0) or 0 for r in group)
            if final_model_errors or repair_errors or repair_timeouts or preserved:
                lines.append(
                    f"| {label} | {final_model_errors} | {repair_errors} | "
                    f"{repair_timeouts} | {preserved} |"
                )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--batch-tag", default=None)
    args = parser.parse_args()
    rows = _load_rows(args.jsonl, args.batch_tag)
    if not rows:
        raise SystemExit("no rows matched")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
