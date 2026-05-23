"""Generate plots from results/runs.jsonl.

Saves PNGs into results/plots/<batch_tag>/ so each batch's view is preserved.

Plots produced (only those that have data):
  1. pass_rate_by_mode_k.png        — bar chart, pass@1 on hidden tests per (mode, k)
  2. baseline_vs_tests.png          — no tests (A) vs public-test feedback (C)
  3. baseline_vs_self_tests.png     — no tests (A) vs model-written tests (D)
  4. baseline_vs_separate_self_tests.png — no tests (A) vs separated self-test loop (D_sep)
  5. baseline_vs_dual_self_tests.png — no tests (A) vs two-model self-test loop (D_dual)
  6. baseline_vs_validated_self_tests.png — no tests (A) vs validated self-test loop (D_val)
  7. pass_rate_vs_iterations.png    — line plot, k-sweep for exec modes
  8. repair_outcomes.png            — hidden-fail to hidden-pass after self-test feedback
  9. overfit_rate_by_mode.png       — visible-pass-but-hidden-fail rate per (mode, k)
  10. tokens_vs_pass.png             — scatter, output tokens vs hidden pass
  11. delta_by_model_size.png       — only when multiple models present
  12. per_task_heatmap.png          — task x mode_k grid of pass/fail
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .analysis import build_ablation_summary

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_rows(
    rows: list[dict],
    model: str | None = None,
    batch_tag: str | None = None,
) -> list[dict]:
    out = rows
    if model:
        out = [r for r in out if r.get("model") == model]
    if batch_tag:
        out = [r for r in out if (r.get("extra") or {}).get("batch_tag") == batch_tag]
    return out


def _label(mode: str, k: int) -> str:
    return f"{mode}/k={k}"


def _group_pass_rate(rows: list[dict]) -> dict[tuple[str, int], tuple[int, int]]:
    """Returns {(mode,k): (passed, total)}."""
    agg: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("passed_hidden") is None:
            continue
        key = (r["mode"], r["k"])
        agg[key][1] += 1
        if r["passed_hidden"]:
            agg[key][0] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}


def plot_pass_rate_by_mode_k(rows: list[dict], out_path: Path) -> None:
    grouped = _group_pass_rate(rows)
    if not grouped:
        return
    keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    labels = [_label(m, k) for m, k in keys]
    rates = [grouped[k][0] / grouped[k][1] if grouped[k][1] else 0.0 for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.8), 4))
    bars = ax.bar(labels, [r * 100 for r in rates], color="#3b82f6")
    ax.set_ylabel("hidden pass rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Hidden test pass rate by mode/k")
    for bar, (m, k) in zip(bars, keys):
        passed, total = grouped[(m, k)]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{passed}/{total}",
            ha="center",
            fontsize=8,
        )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_baseline_vs_tests(rows: list[dict], out_path: Path) -> None:
    """Simple headline chart: Mode A/k=1 vs Mode C at the largest available k.

    A is the no-test/no-execution baseline. C is public tests + execution
    feedback. This is the cleanest first read of the hypothesis.
    """
    scored = [r for r in rows if r.get("passed_hidden") is not None]
    if not scored:
        return

    comparisons = []
    for model in sorted({r["model"] for r in scored}):
        sub = [r for r in scored if r["model"] == model]
        grouped = _group_pass_rate(sub)
        baseline = grouped.get(("A", 1))
        c_keys = sorted((key for key in grouped if key[0] == "C"), key=lambda x: x[1])
        if not baseline or not c_keys:
            continue
        test_key = c_keys[-1]
        comparisons.append((model, baseline, test_key, grouped[test_key]))
    if not comparisons:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(comparisons) * 2.2), 4.5))
    colors = ["#94a3b8", "#2563eb"]

    if len(comparisons) == 1:
        model, baseline, test_key, with_tests = comparisons[0]
        values = [
            baseline[0] / baseline[1] * 100 if baseline[1] else 0,
            with_tests[0] / with_tests[1] * 100 if with_tests[1] else 0,
        ]
        labels = ["No tests\nA/k=1", f"With tests\nC/k={test_key[1]}"]
        bars = ax.bar(labels, values, color=colors, width=0.55)
        for bar, (passed, total) in zip(bars, [baseline, with_tests]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{passed}/{total}",
                ha="center",
                fontsize=9,
            )
        delta = values[1] - values[0]
        ax.text(
            0.5,
            max(values) + 8,
            f"lift: {delta:+.1f} percentage points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_title(f"No tests vs public-test feedback\n{model}")
    else:
        x = list(range(len(comparisons)))
        width = 0.36
        baseline_rates = []
        with_test_rates = []
        for _, baseline, _, with_tests in comparisons:
            baseline_rates.append(baseline[0] / baseline[1] * 100 if baseline[1] else 0)
            with_test_rates.append(with_tests[0] / with_tests[1] * 100 if with_tests[1] else 0)
        bars_a = ax.bar([i - width / 2 for i in x], baseline_rates, width, label="No tests (A/k=1)", color=colors[0])
        bars_c = ax.bar([i + width / 2 for i in x], with_test_rates, width, label="With tests (C/k=max)", color=colors[1])
        for bars, pairs in ((bars_a, [c[1] for c in comparisons]), (bars_c, [c[3] for c in comparisons])):
            for bar, (passed, total) in zip(bars, pairs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{passed}/{total}",
                    ha="center",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in comparisons], rotation=20, ha="right")
        ax.legend()
        ax.set_title("No tests vs public-test feedback")

    ax.set_ylabel("hidden pass rate on benchmark (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_baseline_vs_self_tests(rows: list[dict], out_path: Path) -> None:
    """Headline chart for the self-test hypothesis: Mode A/k=1 vs Mode D/k=max.

    A is one-shot code with no tests. D is model-written tests plus execution
    feedback, iterating until the model's own tests pass or the budget ends.
    """
    _plot_baseline_vs_mode(
        rows=rows,
        out_path=out_path,
        compared_mode="D",
        compared_label="With self-tests",
        compared_detail="D/k=max",
        title="No tests vs model-written self-test loop",
        color="#10b981",
    )


def plot_baseline_vs_separate_self_tests(rows: list[dict], out_path: Path) -> None:
    """Headline chart for separated self-tests: Mode A/k=1 vs Mode D_sep/k=max."""
    _plot_baseline_vs_mode(
        rows=rows,
        out_path=out_path,
        compared_mode="D_sep",
        compared_label="With separate self-tests",
        compared_detail="D_sep/k=max",
        title="No tests vs separated self-test loop",
        color="#14b8a6",
    )


def plot_baseline_vs_dual_self_tests(rows: list[dict], out_path: Path) -> None:
    """Headline chart for two-model self-tests: Mode A/k=1 vs Mode D_dual/k=max."""
    _plot_baseline_vs_mode(
        rows=rows,
        out_path=out_path,
        compared_mode="D_dual",
        compared_label="With dual self-tests",
        compared_detail="D_dual/k=max",
        title="No tests vs two-model self-test loop",
        color="#0f766e",
    )


def plot_baseline_vs_validated_self_tests(rows: list[dict], out_path: Path) -> None:
    """Headline chart for validated self-tests: Mode A/k=1 vs Mode D_val/k=max."""
    _plot_baseline_vs_mode(
        rows=rows,
        out_path=out_path,
        compared_mode="D_val",
        compared_label="With validated self-tests",
        compared_detail="D_val/k=max",
        title="No tests vs validated self-test loop",
        color="#15803d",
    )


def _plot_baseline_vs_mode(
    rows: list[dict],
    out_path: Path,
    compared_mode: str,
    compared_label: str,
    compared_detail: str,
    title: str,
    color: str,
) -> None:
    scored = [r for r in rows if r.get("passed_hidden") is not None]
    if not scored:
        return

    comparisons = []
    for model in sorted({r["model"] for r in scored}):
        sub = [r for r in scored if r["model"] == model]
        grouped = _group_pass_rate(sub)
        baseline = grouped.get(("A", 1))
        compared_keys = sorted((key for key in grouped if key[0] == compared_mode), key=lambda x: x[1])
        if not baseline or not compared_keys:
            continue
        compared_key = compared_keys[-1]
        comparisons.append((model, baseline, compared_key, grouped[compared_key]))
    if not comparisons:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(comparisons) * 2.2), 4.5))
    colors = ["#94a3b8", color]

    if len(comparisons) == 1:
        model, baseline, compared_key, compared = comparisons[0]
        values = [
            baseline[0] / baseline[1] * 100 if baseline[1] else 0,
            compared[0] / compared[1] * 100 if compared[1] else 0,
        ]
        labels = ["No tests\nA/k=1", f"{compared_label}\n{compared_mode}/k={compared_key[1]}"]
        bars = ax.bar(labels, values, color=colors, width=0.55)
        for bar, (passed, total) in zip(bars, [baseline, compared]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{passed}/{total}",
                ha="center",
                fontsize=9,
            )
        delta = values[1] - values[0]
        ax.text(
            0.5,
            max(values) + 8,
            f"lift: {delta:+.1f} percentage points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_title(f"{title}\n{model}")
    else:
        x = list(range(len(comparisons)))
        width = 0.36
        baseline_rates = []
        compared_rates = []
        for _, baseline, _, compared in comparisons:
            baseline_rates.append(baseline[0] / baseline[1] * 100 if baseline[1] else 0)
            compared_rates.append(compared[0] / compared[1] * 100 if compared[1] else 0)
        bars_a = ax.bar([i - width / 2 for i in x], baseline_rates, width, label="No tests (A/k=1)", color=colors[0])
        bars_b = ax.bar([i + width / 2 for i in x], compared_rates, width, label=f"{compared_label} ({compared_detail})", color=colors[1])
        for bars, pairs in ((bars_a, [c[1] for c in comparisons]), (bars_b, [c[3] for c in comparisons])):
            for bar, (passed, total) in zip(bars, pairs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{passed}/{total}",
                    ha="center",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in comparisons], rotation=20, ha="right")
        ax.legend()
        ax.set_title(title)

    ax.set_ylabel("hidden pass rate on benchmark (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pass_rate_vs_iterations(rows: list[dict], out_path: Path) -> None:
    grouped = _group_pass_rate(rows)
    by_mode: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (mode, k), (p, t) in grouped.items():
        if t == 0:
            continue
        by_mode[mode].append((k, p / t * 100))
    if not by_mode:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {
        "A": "#94a3b8",
        "B": "#f59e0b",
        "C": "#3b82f6",
        "D": "#10b981",
        "D_sep": "#14b8a6",
        "D_dual": "#0f766e",
        "D_val": "#15803d",
        "E": "#a855f7",
    }
    for mode, pts in sorted(by_mode.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=f"mode {mode}", color=colors.get(mode))
    ax.set_xlabel("k (max iterations)")
    ax.set_ylabel("hidden pass rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Pass rate vs iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
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
    buckets = ["fixed", "already pass", "not fixed", "regressed", "unknown"]
    counts = {bucket: 0 for bucket in buckets}
    for row in rows:
        bucket = _repair_bucket(row)
        if bucket is not None:
            counts[bucket] += 1
    if not any(counts.values()):
        return

    labels = [bucket for bucket in buckets if counts[bucket]]
    values = [counts[bucket] for bucket in labels]
    colors = {
        "fixed": "#10b981",
        "already pass": "#60a5fa",
        "not fixed": "#ef4444",
        "regressed": "#f59e0b",
        "unknown": "#94a3b8",
    }
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    bars = ax.bar(labels, values, color=[colors[label] for label in labels])
    ax.set_ylabel("run count")
    ax.set_title("Did self-test feedback repair hidden failures?")
    total = sum(values)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{value}/{total}",
            ha="center",
            fontsize=9,
        )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overfit_rate(rows: list[dict], out_path: Path) -> None:
    agg: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("passed_hidden") is None:
            continue
        key = (r["mode"], r["k"])
        visible_pass = _row_visible_passed(r)
        agg[key][1] += 1
        if visible_pass and not r["passed_hidden"]:
            agg[key][0] += 1
    keys = [k for k in sorted(agg.keys()) if agg[k][1] > 0]
    if not keys:
        return
    labels = [_label(m, k) for m, k in keys]
    rates = [agg[k][0] / agg[k][1] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.8), 4))
    bars = ax.bar(labels, rates, color="#ef4444")
    ax.set_ylabel("overfit rate (%)")
    ax.set_ylim(0, max(100, max(rates) + 5))
    ax.set_title("Overfit: visible tests pass, hidden tests fail")
    for bar, k in zip(bars, keys):
        bad, tot = agg[k]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bad}/{tot}",
            ha="center",
            fontsize=8,
        )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _row_visible_passed(row: dict) -> bool:
    mode = row.get("mode")
    required = []
    if mode in ("C", "E"):
        required.append(row.get("passed_public"))
    if mode in ("D", "D_sep", "D_dual", "D_val", "E"):
        required.append(row.get("passed_self"))
    return bool(required) and all(value is True for value in required)


def plot_tokens_vs_pass(rows: list[dict], out_path: Path) -> None:
    pts_pass_x = []
    pts_pass_y = []
    pts_fail_x = []
    pts_fail_y = []
    for r in rows:
        if r.get("passed_hidden") is None:
            continue
        x = r.get("tokens_out", 0)
        y_mode_k = f"{r['mode']}/k={r['k']}"
        if r["passed_hidden"]:
            pts_pass_x.append(x)
            pts_pass_y.append(y_mode_k)
        else:
            pts_fail_x.append(x)
            pts_fail_y.append(y_mode_k)
    if not pts_pass_x and not pts_fail_x:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(pts_pass_x, pts_pass_y, color="#10b981", label="hidden PASS", alpha=0.7)
    ax.scatter(pts_fail_x, pts_fail_y, color="#ef4444", label="hidden FAIL", alpha=0.7, marker="x")
    ax.set_xlabel("output tokens (sum across iterations)")
    ax.set_ylabel("mode/k")
    ax.set_title("Tokens spent vs outcome")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_by_model_size(rows: list[dict], out_path: Path) -> None:
    """Delta (mode X k=max - mode A k=1) per model.

    Useful once multiple models have data in the JSONL.
    """
    models = sorted({r["model"] for r in rows})
    if len(models) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in ["C", "D", "D_sep", "D_dual", "D_val", "E"]:
        xs, ys = [], []
        for m in models:
            sub = [r for r in rows if r["model"] == m]
            grp = _group_pass_rate(sub)
            a_key = ("A", 1)
            if a_key not in grp:
                continue
            a_rate = grp[a_key][0] / grp[a_key][1] if grp[a_key][1] else 0.0
            mode_keys = sorted([k for k in grp if k[0] == mode], key=lambda x: x[1])
            if not mode_keys:
                continue
            best_k = mode_keys[-1]
            p, t = grp[best_k]
            if t == 0:
                continue
            xs.append(m)
            ys.append((p / t - a_rate) * 100)
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{mode} - A")
    ax.set_xlabel("model")
    ax.set_ylabel("delta hidden pass rate (pp)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Lift from loop vs one-shot, by model")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_task_heatmap(rows: list[dict], out_path: Path) -> None:
    tasks = sorted({r["task_id"] for r in rows})
    cells: dict[tuple[str, str], str] = {}
    mode_ks = sorted({(r["mode"], r["k"]) for r in rows}, key=lambda x: (x[0], x[1]))
    for r in rows:
        if r.get("passed_hidden") is None:
            continue
        col = _label(r["mode"], r["k"])
        cells[(r["task_id"], col)] = "P" if r["passed_hidden"] else "F"
    if not cells:
        return
    cols = [_label(m, k) for m, k in mode_ks]
    grid = []
    for t in tasks:
        row = []
        for c in cols:
            v = cells.get((t, c))
            row.append(1 if v == "P" else (0 if v == "F" else -1))
        grid.append(row)
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.7), max(3, len(tasks) * 0.4)))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    for i, t in enumerate(tasks):
        for j, c in enumerate(cols):
            v = cells.get((t, c))
            if v is not None:
                ax.text(j, i, v, ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Per-task outcomes (P=hidden pass, F=fail, blank=not run)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_all(
    jsonl_path: Path,
    out_dir: Path,
    model: str | None = None,
    batch_tag: str | None = None,
) -> list[Path]:
    rows = load(jsonl_path)
    rows = filter_rows(rows, model=model, batch_tag=batch_tag)
    if not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    funcs = [
        ("pass_rate_by_mode_k.png", plot_pass_rate_by_mode_k),
        ("baseline_vs_tests.png", plot_baseline_vs_tests),
        ("baseline_vs_self_tests.png", plot_baseline_vs_self_tests),
        ("baseline_vs_separate_self_tests.png", plot_baseline_vs_separate_self_tests),
        ("baseline_vs_dual_self_tests.png", plot_baseline_vs_dual_self_tests),
        ("baseline_vs_validated_self_tests.png", plot_baseline_vs_validated_self_tests),
        ("pass_rate_vs_iterations.png", plot_pass_rate_vs_iterations),
        ("repair_outcomes.png", plot_repair_outcomes),
        ("overfit_rate_by_mode.png", plot_overfit_rate),
        ("tokens_vs_pass.png", plot_tokens_vs_pass),
        ("delta_by_model_size.png", plot_delta_by_model_size),
        ("per_task_heatmap.png", plot_per_task_heatmap),
    ]
    for name, fn in funcs:
        p = out_dir / name
        fn(rows, p)
        if p.exists():
            plots.append(p)

    # Write a summary.json so the batch dir is self-describing.
    repair_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        bucket = _repair_bucket(row)
        if bucket is not None:
            repair_counts[bucket] += 1
    summary = {
        "n_rows": len(rows),
        "model_filter": model,
        "batch_tag": batch_tag,
        "models": sorted({r["model"] for r in rows}),
        "tasks": sorted({r["task_id"] for r in rows}),
        "modes": sorted({r["mode"] for r in rows}),
        "ks": sorted({r["k"] for r in rows}),
        "hidden_pass_count": sum(1 for r in rows if r.get("passed_hidden")),
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

    p = argparse.ArgumentParser()
    p.add_argument("--log", default="results/runs.jsonl")
    p.add_argument("--out", default=None, help="output dir; defaults to results/plots/<timestamp>")
    p.add_argument("--model", default=None, help="filter to one model")
    p.add_argument("--batch-tag", default=None, help="filter to one batch tag")
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else Path("results/plots") / default_batch_tag()
    plots = make_all(Path(args.log), out_dir, model=args.model, batch_tag=args.batch_tag)
    if not plots:
        print("no data to plot")
        return 1
    print(f"wrote {len(plots)} plots to {out_dir}")
    for p_ in plots:
        print(f"  {p_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
