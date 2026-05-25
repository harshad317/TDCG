"""Ablation summaries for code-generation benchmark runs."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def _row_key(row: dict) -> tuple[str, int, str]:
    return (row["model"], int(row.get("seed", 0)), row["task_id"])


def _scored(row: dict) -> bool:
    return row.get("passed_hidden") is not None


def self_test_confusion(rows: list[dict]) -> dict[str, Any]:
    """Measure whether model-written tests agree with hidden tests.

    For self-test modes, `passed_self=True` means the candidate was accepted by
    the model-written tests. `passed_hidden=True` is the real benchmark result.
    """
    counts = {
        "accepted_good": 0,  # self pass, hidden pass
        "missed_bug": 0,  # self pass, hidden fail
        "false_alarm": 0,  # self fail, hidden pass
        "caught_bug": 0,  # self fail, hidden fail
        "not_scored": 0,
    }
    by_mode_k: dict[str, dict[str, int]] = defaultdict(lambda: {key: 0 for key in counts})

    for row in rows:
        if row.get("mode") not in ("D", "D_sep", "D_dual", "D_val", "E"):
            continue
        label = f"{row['mode']}/k={row['k']}"
        self_passed = row.get("passed_self")
        hidden_passed = row.get("passed_hidden")
        if self_passed is None or hidden_passed is None:
            bucket = "not_scored"
        elif self_passed is True and hidden_passed is True:
            bucket = "accepted_good"
        elif self_passed is True and hidden_passed is False:
            bucket = "missed_bug"
        elif self_passed is False and hidden_passed is True:
            bucket = "false_alarm"
        else:
            bucket = "caught_bug"
        counts[bucket] += 1
        by_mode_k[label][bucket] += 1

    useful_total = sum(counts[k] for k in ("accepted_good", "missed_bug", "false_alarm", "caught_bug"))
    bug_total = counts["missed_bug"] + counts["caught_bug"]
    hidden_pass_total = counts["accepted_good"] + counts["false_alarm"]
    return {
        "counts": counts,
        "by_mode_k": dict(sorted(by_mode_k.items())),
        "self_test_precision_when_passed": _safe_rate(
            counts["accepted_good"],
            counts["accepted_good"] + counts["missed_bug"],
        ),
        "bug_detection_rate": _safe_rate(counts["caught_bug"], bug_total),
        "false_alarm_rate_on_hidden_pass": _safe_rate(counts["false_alarm"], hidden_pass_total),
        "scored_total": useful_total,
    }


def paired_ablation(rows: list[dict]) -> dict[str, Any]:
    """Pair A/k=1 against each executable/test mode by model, seed, and task."""
    scored = [row for row in rows if _scored(row)]
    baseline: dict[tuple[str, int, str], dict] = {}
    for row in scored:
        if row.get("mode") == "A" and int(row.get("k", 0)) == 1:
            baseline[_row_key(row)] = row

    by_mode_k: dict[str, dict[str, Any]] = {}
    compared_rows = [
        row
        for row in scored
        if row.get("mode") in ("C", "D", "D_sep", "D_dual", "D_val", "E", "P_select")
    ]
    for row in compared_rows:
        base = baseline.get(_row_key(row))
        if base is None:
            continue
        label = f"{row['mode']}/k={row['k']}"
        bucket = by_mode_k.setdefault(
            label,
            {
                "total": 0,
                "helped": 0,  # A failed, compared mode passed
                "hurt": 0,  # A passed, compared mode failed
                "both_pass": 0,
                "both_fail": 0,
                "baseline_passes": 0,
                "compared_passes": 0,
                "examples_helped": [],
                "examples_hurt": [],
            },
        )
        base_pass = bool(base["passed_hidden"])
        compared_pass = bool(row["passed_hidden"])
        bucket["total"] += 1
        bucket["baseline_passes"] += int(base_pass)
        bucket["compared_passes"] += int(compared_pass)
        if not base_pass and compared_pass:
            bucket["helped"] += 1
            _append_example(bucket["examples_helped"], row)
        elif base_pass and not compared_pass:
            bucket["hurt"] += 1
            _append_example(bucket["examples_hurt"], row)
        elif base_pass and compared_pass:
            bucket["both_pass"] += 1
        else:
            bucket["both_fail"] += 1

    for bucket in by_mode_k.values():
        total = bucket["total"]
        bucket["baseline_pass_rate"] = _safe_rate(bucket["baseline_passes"], total)
        bucket["compared_pass_rate"] = _safe_rate(bucket["compared_passes"], total)
        bucket["net_lift_pp"] = (
            None
            if total == 0
            else round((bucket["compared_passes"] - bucket["baseline_passes"]) / total * 100, 2)
        )

    return {
        "baseline": "A/k=1",
        "by_mode_k": dict(sorted(by_mode_k.items())),
    }


def repair_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        extra = row.get("extra") or {}
        if extra.get("score_hidden_each_iter") is not True:
            continue
        if row.get("mode") not in ("D", "D_sep", "D_dual", "D_val", "E"):
            continue
        initial = extra.get("initial_hidden_pass")
        final = row.get("passed_hidden")
        if extra.get("fixed_by_self_tests") is True:
            bucket = "fixed"
        elif initial is True and final is True:
            bucket = "already_pass"
        elif initial is False and final is False:
            bucket = "not_fixed"
        elif initial is True and final is False:
            bucket = "regressed"
        else:
            bucket = "unknown"
        counts[bucket] += 1
    return dict(sorted(counts.items()))


def build_ablation_summary(rows: list[dict]) -> dict[str, Any]:
    return {
        "paired_ablation": paired_ablation(rows),
        "self_test_confusion": self_test_confusion(rows),
        "repair_counts": repair_counts(rows),
    }


def format_ablation_summary(summary: dict[str, Any]) -> str:
    lines = ["Ablation summary:"]
    paired = summary.get("paired_ablation", {}).get("by_mode_k", {})
    if paired:
        lines.append("  Final hidden outcome, paired against A/k=1:")
        for label, stats in paired.items():
            total = stats["total"]
            lines.append(
                "  "
                f"{label}: helped={stats['helped']} hurt={stats['hurt']} "
                f"both_pass={stats['both_pass']} both_fail={stats['both_fail']} "
                f"net_lift={stats['net_lift_pp']:+.2f}pp n={total}"
            )
    else:
        lines.append("  Final hidden outcome: no A/k=1 pairs found.")

    confusion = summary.get("self_test_confusion", {}).get("counts", {})
    if confusion:
        lines.append(
            "  Self-test vs hidden: "
            f"accepted_good={confusion.get('accepted_good', 0)} "
            f"missed_bug={confusion.get('missed_bug', 0)} "
            f"false_alarm={confusion.get('false_alarm', 0)} "
            f"caught_bug={confusion.get('caught_bug', 0)}"
        )

    repair = summary.get("repair_counts", {})
    if repair:
        repair_text = " ".join(f"{key}={value}" for key, value in repair.items())
        lines.append(f"  Repair trace: {repair_text}")
    return "\n".join(lines)


def _append_example(examples: list[dict], row: dict, limit: int = 10) -> None:
    if len(examples) >= limit:
        return
    examples.append(
        {
            "task_id": row["task_id"],
            "model": row["model"],
            "seed": row.get("seed", 0),
            "mode": row["mode"],
            "k": row["k"],
        }
    )


def _safe_rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(num / den, 4)
