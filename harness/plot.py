"""Generate useful plots from JSONL benchmark logs.

The plotting layer is intentionally coverage-aware:
- append-only logs are deduplicated to the latest row per task/mode/k/seed;
- rerun gains compare the first and latest row for each run key;
- incomplete baselines are not plotted as headline comparisons.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from .analysis import build_ablation_summary

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


MODE_ORDER = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "D_sep": 4,
    "D_dual": 5,
    "D_val": 6,
    "E": 7,
    "P_select": 8,
    "X_codex": 9,
    "X_claude": 10,
}
MODE_COLORS = {
    "A": "#94a3b8",
    "B": "#f59e0b",
    "C": "#3b82f6",
    "D": "#10b981",
    "D_sep": "#14b8a6",
    "D_dual": "#0f766e",
    "D_val": "#15803d",
    "E": "#a855f7",
    "P_select": "#64748b",
    "X_codex": "#111827",
    "X_claude": "#f97316",
}


def load(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _filter_raw_rows(
    rows: list[dict],
    model: str | None = None,
    batch_tag: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    out = rows
    if model:
        out = [r for r in out if r.get("model") == model]
    if batch_tag:
        out = [r for r in out if (r.get("extra") or {}).get("batch_tag") == batch_tag]
    if seed is not None:
        out = [r for r in out if int(r.get("seed", 0)) == seed]
    return out


def filter_rows(
    rows: list[dict],
    model: str | None = None,
    batch_tag: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    return _latest_rows_by_run_key(_filter_raw_rows(rows, model=model, batch_tag=batch_tag, seed=seed))


def _run_key(row: dict) -> tuple | None:
    try:
        extra = row.get("extra") or {}
        batch = extra.get("batch_tag")
        if batch is None:
            return None
        return (
            batch,
            row["model"],
            row["task_id"],
            row["mode"],
            int(row["k"]),
            int(row.get("seed", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _latest_rows_by_run_key(rows: list[dict]) -> list[dict]:
    latest: dict[tuple, dict] = {}
    latest_index: dict[tuple, int] = {}
    unkeyed: list[tuple[int, dict]] = []
    for index, row in enumerate(rows):
        key = _run_key(row)
        if key is None:
            unkeyed.append((index, row))
            continue
        latest[key] = row
        latest_index[key] = index

    keyed = [(latest_index[key], row) for key, row in latest.items()]
    return [row for _, row in sorted(unkeyed + keyed, key=lambda item: item[0])]


def _first_and_latest_by_run_key(rows: list[dict]) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    first: dict[tuple, dict] = {}
    latest: dict[tuple, dict] = {}
    for row in rows:
        key = _run_key(row)
        if key is None:
            continue
        first.setdefault(key, row)
        latest[key] = row
    return first, latest


def _task_sort_key(task_id: str) -> tuple[int, str]:
    try:
        return int(task_id.rsplit("_", 1)[1]), task_id
    except (IndexError, ValueError):
        return 10**9, task_id


def _mode_k_sort_key(key: tuple[str, int]) -> tuple[int, int, str]:
    mode, k = key
    return MODE_ORDER.get(mode, 99), int(k), mode


def _label(mode: str, k: int) -> str:
    return f"{mode}/k={k}"


def _group_pass_rate(rows: list[dict]) -> dict[tuple[str, int], tuple[int, int]]:
    agg: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row.get("passed_hidden") is None:
            continue
        key = (row["mode"], int(row["k"]))
        agg[key][1] += 1
        if row.get("passed_hidden") is True:
            agg[key][0] += 1
    return {key: (value[0], value[1]) for key, value in agg.items()}


def _model_mode_k_sort_key(key: tuple[str, str, int]) -> tuple[str, int, int, str]:
    model, mode, k = key
    return model.lower(), MODE_ORDER.get(mode, 99), int(k), mode


def plot_pass_rate_by_model_mode_k(rows: list[dict], out_path: Path) -> None:
    if len({row["model"] for row in rows}) < 2:
        return
    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row.get("passed_hidden") is None:
            continue
        key = (row["model"], row["mode"], int(row["k"]))
        grouped[key][1] += 1
        grouped[key][0] += int(row.get("passed_hidden") is True)
    if not grouped:
        return

    keys = sorted(grouped, key=_model_mode_k_sort_key)
    labels = [f"{model} | {_label(mode, k)}" for model, mode, k in keys]
    rates = [grouped[key][0] / grouped[key][1] * 100 for key in keys]
    text = [
        f"{passed}/{total} ({passed / total * 100:.1f}%)"
        for passed, total in (grouped[key] for key in keys)
    ]

    fig, ax = plt.subplots(figsize=(11, max(4.2, len(keys) * 0.5)))
    y = list(range(len(keys)))
    bars = ax.barh(
        y,
        rates,
        color=[MODE_COLORS.get(mode, "#3b82f6") for _, mode, _ in keys],
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("hidden pass rate (%)")
    ax.set_xlim(0, 108)
    ax.set_title("Latest hidden pass rate by model and mode/k")
    ax.grid(True, axis="x", alpha=0.25)
    _bar_text(ax, bars, text, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _row_visible_passed(row: dict) -> bool:
    mode = row.get("mode")
    required: list[bool | None] = []
    if mode in ("C", "D_val", "E", "P_select") or str(mode).startswith("X_"):
        required.append(row.get("passed_public"))
    if mode in ("D", "D_sep", "D_dual", "D_val", "E"):
        required.append(row.get("passed_self"))
    return bool(required) and all(value is True for value in required)


def _bar_text(ax, bars, labels: list[str], *, fontsize: int = 9) -> None:
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=fontsize,
        )


def plot_pass_rate_by_mode_k(rows: list[dict], out_path: Path) -> None:
    grouped = _group_pass_rate(rows)
    if not grouped:
        return
    keys = sorted(grouped, key=_mode_k_sort_key)
    labels = [_label(*key) for key in keys]
    rates = [grouped[key][0] / grouped[key][1] * 100 for key in keys]
    text = [
        f"{passed}/{total} ({passed / total * 100:.1f}%)"
        for passed, total in (grouped[key] for key in keys)
    ]

    fig, ax = plt.subplots(figsize=(9, max(3.8, len(keys) * 0.55)))
    y = list(range(len(keys)))
    bars = ax.barh(
        y,
        rates,
        color=[MODE_COLORS.get(mode, "#3b82f6") for mode, _ in keys],
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("hidden pass rate (%)")
    ax.set_xlim(0, 108)
    ax.set_title("Latest hidden pass rate by mode/k")
    ax.grid(True, axis="x", alpha=0.25)
    _bar_text(ax, bars, text)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_failure_count_by_mode(rows: list[dict], out_path: Path) -> None:
    grouped: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"pass": 0, "fail": 0, "timeout": 0, "unscored": 0}
    )
    for row in rows:
        key = (row["mode"], int(row["k"]))
        if row.get("passed_hidden") is True:
            grouped[key]["pass"] += 1
        elif row.get("passed_hidden") is False:
            if (row.get("extra") or {}).get("hidden_timed_out") is True:
                grouped[key]["timeout"] += 1
            else:
                grouped[key]["fail"] += 1
        else:
            grouped[key]["unscored"] += 1
    if not grouped:
        return

    keys = sorted(grouped, key=_mode_k_sort_key)
    labels = [_label(*key) for key in keys]
    y = list(range(len(keys)))
    pass_counts = [grouped[key]["pass"] for key in keys]
    fail_counts = [grouped[key]["fail"] for key in keys]
    timeout_counts = [grouped[key]["timeout"] for key in keys]
    unscored_counts = [grouped[key]["unscored"] for key in keys]

    fig, ax = plt.subplots(figsize=(9, max(3.8, len(keys) * 0.55)))
    left = [0] * len(keys)
    ax.barh(y, pass_counts, left=left, color="#16a34a", label="hidden pass")
    left = [a + b for a, b in zip(left, pass_counts)]
    ax.barh(y, fail_counts, left=left, color="#ef4444", label="hidden fail")
    left = [a + b for a, b in zip(left, fail_counts)]
    ax.barh(y, timeout_counts, left=left, color="#f97316", label="hidden timeout")
    left = [a + b for a, b in zip(left, timeout_counts)]
    ax.barh(y, unscored_counts, left=left, color="#94a3b8", label="unscored")
    totals = [sum(grouped[key].values()) for key in keys]
    for idx, total in enumerate(totals):
        bad = fail_counts[idx] + timeout_counts[idx] + unscored_counts[idx]
        ax.text(total + 0.8, idx, f"{bad} not pass / {total}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("run count")
    ax.set_title("Pass/fail/timeout counts by mode/k")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rerun_gain_by_mode(raw_rows: list[dict], out_path: Path) -> None:
    first, latest = _first_and_latest_by_run_key(raw_rows)
    duplicated_keys = {key for key in latest if key in first and latest[key] is not first[key]}
    if not duplicated_keys:
        return

    grouped: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"first_pass": 0, "latest_pass": 0, "total": 0}
    )
    for key in duplicated_keys:
        first_row = first[key]
        latest_row = latest[key]
        if first_row.get("passed_hidden") is None or latest_row.get("passed_hidden") is None:
            continue
        group_key = (latest_row["mode"], int(latest_row["k"]))
        grouped[group_key]["total"] += 1
        grouped[group_key]["first_pass"] += int(first_row.get("passed_hidden") is True)
        grouped[group_key]["latest_pass"] += int(latest_row.get("passed_hidden") is True)
    grouped = {key: value for key, value in grouped.items() if value["total"]}
    if not grouped:
        return

    keys = sorted(grouped, key=_mode_k_sort_key)
    labels = [_label(*key) for key in keys]
    first_rates = [
        grouped[key]["first_pass"] / grouped[key]["total"] * 100
        for key in keys
    ]
    latest_rates = [
        grouped[key]["latest_pass"] / grouped[key]["total"] * 100
        for key in keys
    ]
    x = list(range(len(keys)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.2), 4.5))
    bars_a = ax.bar([i - width / 2 for i in x], first_rates, width, color="#cbd5e1", label="first row")
    bars_b = ax.bar([i + width / 2 for i in x], latest_rates, width, color="#2563eb", label="latest row")
    for i, key in enumerate(keys):
        data = grouped[key]
        delta = data["latest_pass"] - data["first_pass"]
        ax.text(
            i,
            max(first_rates[i], latest_rates[i]) + 3,
            f"{delta:+d}/{data['total']}",
            ha="center",
            fontsize=9,
            fontweight="bold" if delta else "normal",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("pass rate on rerun slice (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Rerun gain by mode/k")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _repair_bucket(row: dict) -> str | None:
    extra = row.get("extra") or {}
    if extra.get("score_hidden_each_iter") is not True:
        return None
    if row.get("mode") not in ("D", "D_sep", "D_dual", "D_val", "E"):
        return None
    initial = extra.get("initial_hidden_pass")
    final = row.get("passed_hidden")
    if extra.get("fixed_by_self_tests") is True:
        return "fixed"
    if initial is True and final is True:
        return "already pass"
    if initial is False and final is False:
        return "not fixed"
    if initial is True and final is False:
        return "regressed"
    return "unknown"


def plot_repair_outcomes(rows: list[dict], out_path: Path) -> None:
    buckets = ["already pass", "fixed", "not fixed", "regressed", "unknown"]
    colors = {
        "already pass": "#60a5fa",
        "fixed": "#10b981",
        "not fixed": "#ef4444",
        "regressed": "#f59e0b",
        "unknown": "#94a3b8",
    }
    grouped: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {bucket: 0 for bucket in buckets}
    )
    for row in rows:
        bucket = _repair_bucket(row)
        if bucket:
            grouped[(row["mode"], int(row["k"]))][bucket] += 1
    grouped = {key: value for key, value in grouped.items() if sum(value.values())}
    if not grouped:
        return

    keys = sorted(grouped, key=_mode_k_sort_key)
    labels = [_label(*key) for key in keys]
    y = list(range(len(keys)))
    left = [0] * len(keys)
    fig, ax = plt.subplots(figsize=(10, max(3.8, len(keys) * 0.55)))
    for bucket in buckets:
        values = [grouped[key][bucket] for key in keys]
        ax.barh(y, values, left=left, color=colors[bucket], label=bucket)
        left = [a + b for a, b in zip(left, values)]
    for idx, key in enumerate(keys):
        total = sum(grouped[key].values())
        fixed = grouped[key]["fixed"]
        ax.text(total + 0.8, idx, f"fixed {fixed}/{total}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("run count")
    ax.set_title("Repair outcome by mode/k")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_self_test_confusion(rows: list[dict], out_path: Path) -> None:
    buckets = ["accepted_good", "missed_bug", "false_alarm", "caught_bug", "not_scored"]
    colors = {
        "accepted_good": "#16a34a",
        "missed_bug": "#dc2626",
        "false_alarm": "#f59e0b",
        "caught_bug": "#2563eb",
        "not_scored": "#94a3b8",
    }
    grouped: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {bucket: 0 for bucket in buckets}
    )
    for row in rows:
        if row.get("mode") not in ("D", "D_sep", "D_dual", "D_val", "E"):
            continue
        key = (row["mode"], int(row["k"]))
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
        grouped[key][bucket] += 1
    grouped = {key: value for key, value in grouped.items() if sum(value.values())}
    if not grouped:
        return

    keys = sorted(grouped, key=_mode_k_sort_key)
    labels = [_label(*key) for key in keys]
    y = list(range(len(keys)))
    left = [0] * len(keys)
    fig, ax = plt.subplots(figsize=(10, max(3.8, len(keys) * 0.55)))
    for bucket in buckets:
        values = [grouped[key][bucket] for key in keys]
        ax.barh(y, values, left=left, color=colors[bucket], label=bucket.replace("_", " "))
        left = [a + b for a, b in zip(left, values)]
    for idx, key in enumerate(keys):
        data = grouped[key]
        accepted = data["accepted_good"] + data["missed_bug"]
        precision = data["accepted_good"] / accepted * 100 if accepted else 0
        ax.text(sum(data.values()) + 0.8, idx, f"self-pass precision {precision:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("run count")
    ax.set_title("Self-test signal vs hidden outcome")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_overfit_rate(rows: list[dict], out_path: Path) -> None:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row.get("passed_hidden") is None:
            continue
        if not _row_visible_passed(row):
            continue
        key = (row["mode"], int(row["k"]))
        grouped[key][1] += 1
        if row.get("passed_hidden") is False:
            grouped[key][0] += 1
    grouped = {key: value for key, value in grouped.items() if value[1]}
    if not grouped:
        return

    keys = sorted(grouped, key=_mode_k_sort_key)
    labels = [_label(*key) for key in keys]
    rates = [grouped[key][0] / grouped[key][1] * 100 for key in keys]
    text = [f"{bad}/{visible}" for bad, visible in (grouped[key] for key in keys)]
    fig, ax = plt.subplots(figsize=(9, max(3.8, len(keys) * 0.55)))
    y = list(range(len(keys)))
    bars = ax.barh(y, rates, color="#ef4444")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("visible-pass rows that fail hidden (%)")
    ax.set_xlim(0, max(20, max(rates) + 8))
    ax.set_title("Overfit rate among visible-passing rows")
    ax.grid(True, axis="x", alpha=0.25)
    _bar_text(ax, bars, text)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_tokens_vs_pass(rows: list[dict], out_path: Path) -> None:
    scored = [row for row in rows if row.get("passed_hidden") is not None]
    if not scored:
        return
    keys = sorted({(row["mode"], int(row["k"])) for row in scored}, key=_mode_k_sort_key)
    y_lookup = {key: idx for idx, key in enumerate(keys)}
    pass_x: list[int] = []
    pass_y: list[float] = []
    fail_x: list[int] = []
    fail_y: list[float] = []
    for idx, row in enumerate(scored):
        key = (row["mode"], int(row["k"]))
        y = y_lookup[key] + ((idx % 7) - 3) * 0.025
        tokens = max(1, int(row.get("tokens_out") or 1))
        if row.get("passed_hidden") is True:
            pass_x.append(tokens)
            pass_y.append(y)
        else:
            fail_x.append(tokens)
            fail_y.append(y)
    fig, ax = plt.subplots(figsize=(9, max(4, len(keys) * 0.55)))
    ax.scatter(pass_x, pass_y, color="#10b981", label="hidden pass", alpha=0.6, s=22)
    ax.scatter(fail_x, fail_y, color="#ef4444", label="hidden fail", alpha=0.75, marker="x", s=30)
    ax.set_xscale("log")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([_label(*key) for key in keys])
    ax.invert_yaxis()
    ax.set_xlabel("output tokens, log scale")
    ax.set_title("Token spend vs hidden outcome")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_remaining_failures_heatmap(rows: list[dict], out_path: Path) -> None:
    preferred_modes = ["D_sep", "D_dual", "D_val", "E"]
    available_cols = [
        (mode, 5)
        for mode in preferred_modes
        if any(row.get("mode") == mode and int(row.get("k", 0)) == 5 for row in rows)
    ]
    if not available_cols:
        available_cols = sorted({(row["mode"], int(row["k"])) for row in rows}, key=_mode_k_sort_key)
    row_by_task_col = {
        (row["task_id"], row["mode"], int(row["k"])): row
        for row in rows
    }
    tasks = sorted(
        {
            row["task_id"]
            for row in rows
            if any(
                row_by_task_col.get((row["task_id"], mode, k), {}).get("passed_hidden") is not True
                for mode, k in available_cols
            )
        },
        key=_task_sort_key,
    )
    if not tasks:
        return
    grid: list[list[int]] = []
    for task in tasks:
        cells = []
        for mode, k in available_cols:
            row = row_by_task_col.get((task, mode, k))
            if row is None or row.get("passed_hidden") is None:
                cells.append(1)
            elif row.get("passed_hidden") is True:
                cells.append(2)
            else:
                cells.append(0)
        grid.append(cells)

    fig, ax = plt.subplots(figsize=(max(6, len(available_cols) * 1.15), max(5, len(tasks) * 0.28)))
    cmap = ListedColormap(["#dc2626", "#cbd5e1", "#16a34a"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(available_cols)))
    ax.set_xticklabels([_label(*key) for key in available_cols], rotation=25, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            label = "P" if value == 2 else ("F" if value == 0 else "-")
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")
    ax.set_title("Remaining failures by task (latest rows)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_pass_rate_vs_iterations(rows: list[dict], out_path: Path) -> None:
    grouped = _group_pass_rate(rows)
    by_mode: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for (mode, k), (passed, total) in grouped.items():
        by_mode[mode].append((k, passed, total))
    by_mode = {
        mode: sorted(points)
        for mode, points in by_mode.items()
        if len(points) >= 2 and len({total for _, _, total in points}) == 1
    }
    if not by_mode:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, points in sorted(by_mode.items(), key=lambda item: MODE_ORDER.get(item[0], 99)):
        xs = [k for k, _, _ in points]
        ys = [passed / total * 100 for _, passed, total in points]
        ax.plot(xs, ys, marker="o", label=mode, color=MODE_COLORS.get(mode))
    ax.set_xlabel("k (max iterations)")
    ax.set_ylabel("hidden pass rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Pass rate vs k (only complete k-sweeps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_delta_by_model_size(rows: list[dict], out_path: Path) -> None:
    models = sorted({row["model"] for row in rows})
    if len(models) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in ["C", "D", "D_sep", "D_dual", "D_val", "E"]:
        xs: list[str] = []
        ys: list[float] = []
        for model in models:
            sub = [row for row in rows if row["model"] == model]
            grouped = _group_pass_rate(sub)
            base = grouped.get(("A", 1))
            mode_keys = sorted([key for key in grouped if key[0] == mode], key=_mode_k_sort_key)
            if not base or not mode_keys:
                continue
            compared = grouped[mode_keys[-1]]
            xs.append(model)
            ys.append((compared[0] / compared[1] - base[0] / base[1]) * 100)
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{mode} - A")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("model")
    ax.set_ylabel("delta hidden pass rate (pp)")
    ax.set_title("Lift from loop vs one-shot, by model")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _coverage_by_mode_k(rows: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "scored": 0, "passed": 0, "failed": 0, "unscored": 0}
    )
    for row in rows:
        key = (row["mode"], int(row["k"]))
        grouped[key]["rows"] += 1
        if row.get("passed_hidden") is None:
            grouped[key]["unscored"] += 1
        else:
            grouped[key]["scored"] += 1
            if row.get("passed_hidden") is True:
                grouped[key]["passed"] += 1
            else:
                grouped[key]["failed"] += 1
    return {
        _label(*key): grouped[key]
        for key in sorted(grouped, key=_mode_k_sort_key)
    }


def _coverage_by_model_mode_k(rows: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "scored": 0, "passed": 0, "failed": 0, "unscored": 0}
    )
    for row in rows:
        key = (row["model"], row["mode"], int(row["k"]))
        grouped[key]["rows"] += 1
        if row.get("passed_hidden") is None:
            grouped[key]["unscored"] += 1
        else:
            grouped[key]["scored"] += 1
            if row.get("passed_hidden") is True:
                grouped[key]["passed"] += 1
            else:
                grouped[key]["failed"] += 1
    return {
        f"{model} | {_label(mode, k)}": grouped[key]
        for key in sorted(grouped, key=_model_mode_k_sort_key)
        for model, mode, k in [key]
    }


def make_all(
    jsonl_path: Path,
    out_dir: Path,
    model: str | None = None,
    batch_tag: str | None = None,
    seed: int | None = None,
) -> list[Path]:
    raw_rows = _filter_raw_rows(load(jsonl_path), model=model, batch_tag=batch_tag, seed=seed)
    rows = _latest_rows_by_run_key(raw_rows)
    if not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)

    for old_png in out_dir.glob("*.png"):
        old_png.unlink()

    plots: list[Path] = []
    funcs: list[tuple[str, Callable[[list[dict], Path], None]]] = [
        ("pass_rate_by_mode_k.png", plot_pass_rate_by_mode_k),
        ("pass_rate_by_model_mode_k.png", plot_pass_rate_by_model_mode_k),
        ("failure_count_by_mode.png", plot_failure_count_by_mode),
        ("repair_outcomes.png", plot_repair_outcomes),
        ("self_test_confusion_by_mode.png", plot_self_test_confusion),
        ("overfit_rate_by_mode.png", plot_overfit_rate),
        ("tokens_vs_pass.png", plot_tokens_vs_pass),
        ("remaining_failures_heatmap.png", plot_remaining_failures_heatmap),
        ("pass_rate_vs_iterations.png", plot_pass_rate_vs_iterations),
        ("delta_by_model_size.png", plot_delta_by_model_size),
    ]
    for name, fn in funcs:
        path = out_dir / name
        fn(rows, path)
        if path.exists():
            plots.append(path)

    rerun_path = out_dir / "rerun_gain_by_mode.png"
    plot_rerun_gain_by_mode(raw_rows, rerun_path)
    if rerun_path.exists():
        plots.append(rerun_path)

    repair_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        bucket = _repair_bucket(row)
        if bucket is not None:
            repair_counts[bucket] += 1
    summary = {
        "raw_rows": len(raw_rows),
        "n_rows": len(rows),
        "model_filter": model,
        "seed_filter": seed,
        "batch_tag": batch_tag,
        "models": sorted({row["model"] for row in rows}),
        "tasks": sorted({row["task_id"] for row in rows}, key=_task_sort_key),
        "modes": sorted({row["mode"] for row in rows}, key=lambda mode: MODE_ORDER.get(mode, 99)),
        "ks": sorted({row["k"] for row in rows}),
        "coverage_by_mode_k": _coverage_by_mode_k(rows),
        "coverage_by_model_mode_k": _coverage_by_model_mode_k(rows),
        "hidden_pass_count": sum(1 for row in rows if row.get("passed_hidden") is True),
        "repair_counts": dict(sorted(repair_counts.items())),
        "ablation": build_ablation_summary(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(summary["ablation"], indent=2) + "\n"
    )
    return plots


def default_batch_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="results/runs.jsonl")
    parser.add_argument("--out", default=None, help="output dir; defaults to results/plots/<timestamp>")
    parser.add_argument("--model", default=None, help="filter to one model")
    parser.add_argument("--batch-tag", default=None, help="filter to one batch tag")
    parser.add_argument("--seed", type=int, default=None, help="filter to one seed")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path("results/plots") / default_batch_tag()
    plots = make_all(Path(args.log), out_dir, model=args.model, batch_tag=args.batch_tag, seed=args.seed)
    if not plots:
        print("no data to plot")
        return 1
    print(f"wrote {len(plots)} plots to {out_dir}")
    for plot in plots:
        print(f"  {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
