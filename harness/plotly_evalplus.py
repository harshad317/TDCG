"""Interactive Plotly charts for EvalPlus benchmark probes."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODE_ORDER = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "D_sep": 4,
    "D_dual": 5,
    "D_val": 6,
    "E": 7,
    "X_codex": 8,
    "X_claude": 9,
}

MODE_COLORS = {
    "A": "#7c8a9f",
    "B": "#e19a2d",
    "C": "#2f80ed",
    "D": "#13a874",
    "D_sep": "#17a2a4",
    "D_dual": "#0d6f65",
    "D_val": "#1f8f4a",
    "E": "#8b5cf6",
    "X_codex": "#111827",
    "X_claude": "#f97316",
}

OUTCOME_COLORS = {
    "pass": "#16a34a",
    "fail": "#dc2626",
    "timeout": "#f97316",
    "unscored": "#94a3b8",
}

REPAIR_COLORS = {
    "already pass": "#60a5fa",
    "fixed": "#10b981",
    "not fixed": "#ef4444",
    "regressed": "#f59e0b",
    "unknown": "#94a3b8",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    title: str
    log_path: Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_rows(rows: Iterable[dict]) -> list[dict]:
    """Keep the latest row per model/task/mode/k/seed, ignoring batch_tag noise."""
    latest: dict[tuple, tuple[int, dict]] = {}
    for index, row in enumerate(rows):
        key = (
            row.get("model"),
            row.get("task_id"),
            row.get("mode"),
            int(row.get("k") or 0),
            int(row.get("seed") or 0),
        )
        latest[key] = (index, row)
    return [row for _, row in sorted(latest.values(), key=lambda item: item[0])]


def mode_key(mode: str, k: int) -> tuple[int, int, str]:
    return MODE_ORDER.get(mode, 99), int(k), mode


def mode_label(mode: str, k: int) -> str:
    return f"{mode}/k={k}"


def common_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Inter, Helvetica, Arial, sans-serif", "size": 13},
        margin={"l": 80, "r": 40, "t": 78, "b": 62},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )


def write_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )


def pass_rate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row.get("passed_hidden") is None:
            continue
        key = (row["mode"], int(row["k"]))
        grouped[key][1] += 1
        grouped[key][0] += int(row.get("passed_hidden") is True)
    out = []
    for mode, k in sorted(grouped, key=lambda key: mode_key(*key)):
        passed, total = grouped[(mode, k)]
        out.append(
            {
                "mode": mode,
                "k": k,
                "label": mode_label(mode, k),
                "passed": passed,
                "total": total,
                "rate": passed / total * 100 if total else 0,
            }
        )
    return out


def plot_pass_rate(rows: list[dict], title: str) -> go.Figure:
    data = pass_rate_rows(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[d["rate"] for d in data],
            y=[d["label"] for d in data],
            orientation="h",
            marker_color=[MODE_COLORS.get(d["mode"], "#3b82f6") for d in data],
            text=[f'{d["passed"]}/{d["total"]} ({d["rate"]:.1f}%)' for d in data],
            textposition="outside",
            hovertemplate=(
                "Mode: %{y}<br>"
                "Hidden pass rate: %{x:.2f}%<br>"
                "Passed/total: %{customdata[0]}/%{customdata[1]}<extra></extra>"
            ),
            customdata=[[d["passed"], d["total"]] for d in data],
        )
    )
    fig.update_xaxes(title="Hidden pass rate (%)", range=[0, 108], ticksuffix="%")
    fig.update_yaxes(title="", autorange="reversed")
    common_layout(fig, title)
    fig.update_layout(height=max(420, 54 * len(data) + 140), showlegend=False)
    return fig


def plot_outcome_counts(rows: list[dict], title: str) -> go.Figure:
    grouped: dict[tuple[str, int], Counter] = defaultdict(Counter)
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
    keys = sorted(grouped, key=lambda key: mode_key(*key))
    labels = [mode_label(*key) for key in keys]

    fig = go.Figure()
    for bucket in ["pass", "fail", "timeout", "unscored"]:
        values = [grouped[key][bucket] for key in keys]
        fig.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                name=bucket,
                marker_color=OUTCOME_COLORS[bucket],
                hovertemplate=f"{bucket}: %{{x}}<br>Mode: %{{y}}<extra></extra>",
            )
        )
    totals = [sum(grouped[key].values()) for key in keys]
    fig.add_trace(
        go.Scatter(
            x=totals,
            y=labels,
            mode="text",
            text=[f"n={total}" for total in totals],
            textposition="middle right",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Latest run count")
    fig.update_yaxes(title="", autorange="reversed")
    common_layout(fig, title)
    fig.update_layout(height=max(420, 54 * len(keys) + 140))
    return fig


def repair_bucket(row: dict) -> str | None:
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


def plot_repair_outcomes(rows: list[dict], title: str) -> go.Figure:
    grouped: dict[tuple[str, int], Counter] = defaultdict(Counter)
    for row in rows:
        bucket = repair_bucket(row)
        if bucket:
            grouped[(row["mode"], int(row["k"]))][bucket] += 1
    keys = sorted(grouped, key=lambda key: mode_key(*key))
    labels = [mode_label(*key) for key in keys]
    fig = go.Figure()
    for bucket in ["already pass", "fixed", "not fixed", "regressed", "unknown"]:
        fig.add_trace(
            go.Bar(
                x=[grouped[key][bucket] for key in keys],
                y=labels,
                orientation="h",
                name=bucket,
                marker_color=REPAIR_COLORS[bucket],
                hovertemplate=f"{bucket}: %{{x}}<br>Mode: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Run count")
    fig.update_yaxes(title="", autorange="reversed")
    common_layout(fig, title)
    fig.update_layout(height=max(420, 54 * len(keys) + 140))
    return fig


def plot_self_test_signal(rows: list[dict], title: str) -> go.Figure:
    buckets = ["accepted_good", "missed_bug", "false_alarm", "caught_bug", "not_scored"]
    colors = {
        "accepted_good": "#16a34a",
        "missed_bug": "#dc2626",
        "false_alarm": "#f59e0b",
        "caught_bug": "#2563eb",
        "not_scored": "#94a3b8",
    }
    grouped: dict[tuple[str, int], Counter] = defaultdict(Counter)
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

    keys = sorted(grouped, key=lambda key: mode_key(*key))
    labels = [mode_label(*key) for key in keys]
    fig = go.Figure()
    for bucket in buckets:
        fig.add_trace(
            go.Bar(
                x=[grouped[key][bucket] for key in keys],
                y=labels,
                orientation="h",
                name=bucket.replace("_", " "),
                marker_color=colors[bucket],
                hovertemplate=f"{bucket.replace('_', ' ')}: %{{x}}<br>Mode: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Run count")
    fig.update_yaxes(title="", autorange="reversed")
    common_layout(fig, title)
    fig.update_layout(height=max(420, 54 * len(keys) + 140))
    return fig


def plot_tokens_vs_wall(rows: list[dict], title: str) -> go.Figure:
    scored = [row for row in rows if row.get("passed_hidden") is not None]
    fig = go.Figure()
    for passed, name, color, symbol in [
        (True, "hidden pass", "#16a34a", "circle"),
        (False, "hidden fail", "#dc2626", "x"),
    ]:
        sub = [row for row in scored if row.get("passed_hidden") is passed]
        fig.add_trace(
            go.Scatter(
                x=[max(1, int(row.get("tokens_out") or 1)) for row in sub],
                y=[float(row.get("wall_time_s") or 0) for row in sub],
                mode="markers",
                name=name,
                marker={"color": color, "symbol": symbol, "size": 8, "opacity": 0.72},
                customdata=[
                    [row.get("task_id"), mode_label(row.get("mode"), int(row.get("k") or 0))]
                    for row in sub
                ],
                hovertemplate=(
                    "Task: %{customdata[0]}<br>"
                    "Mode: %{customdata[1]}<br>"
                    "Output tokens: %{x}<br>"
                    "Wall time: %{y:.1f}s<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(title="Output tokens (log scale)", type="log")
    fig.update_yaxes(title="Wall time (s)")
    common_layout(fig, title)
    fig.update_layout(height=540)
    return fig


def plot_remaining_failures(rows: list[dict], title: str) -> go.Figure:
    preferred = ["D_sep", "D_dual", "D_val", "E"]
    cols = [
        (mode, 5)
        for mode in preferred
        if any(row.get("mode") == mode and int(row.get("k") or 0) == 5 for row in rows)
    ]
    if not cols:
        cols = sorted({(row["mode"], int(row["k"])) for row in rows}, key=lambda key: mode_key(*key))

    by_task_col = {(row["task_id"], row["mode"], int(row["k"])): row for row in rows}
    tasks = sorted(
        {
            row["task_id"]
            for row in rows
            if any(
                by_task_col.get((row["task_id"], mode, k), {}).get("passed_hidden") is not True
                for mode, k in cols
            )
        },
        key=task_key,
    )
    z = []
    text = []
    for task in tasks:
        zrow = []
        trow = []
        for mode, k in cols:
            row = by_task_col.get((task, mode, k))
            if row is None or row.get("passed_hidden") is None:
                zrow.append(1)
                trow.append("unscored")
            elif row.get("passed_hidden") is True:
                zrow.append(2)
                trow.append("pass")
            else:
                zrow.append(0)
                trow.append("fail")
        z.append(zrow)
        text.append(trow)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[mode_label(*col) for col in cols],
            y=tasks,
            text=text,
            colorscale=[
                [0.0, "#dc2626"],
                [0.32, "#dc2626"],
                [0.33, "#cbd5e1"],
                [0.66, "#cbd5e1"],
                [0.67, "#16a34a"],
                [1.0, "#16a34a"],
            ],
            showscale=False,
            hovertemplate="Task: %{y}<br>Mode: %{x}<br>Status: %{text}<extra></extra>",
        )
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="", autorange="reversed")
    common_layout(fig, title)
    fig.update_layout(height=max(520, min(1800, 20 * len(tasks) + 160)))
    return fig


def task_key(task_id: str) -> tuple[int, str]:
    try:
        return int(task_id.rsplit("_", 1)[1]), task_id
    except (IndexError, ValueError):
        return 10**9, task_id


def benchmark_summary(rows: list[dict], raw_rows: list[dict]) -> dict:
    rates = pass_rate_rows(rows)
    return {
        "raw_rows": len(raw_rows),
        "latest_rows": len(rows),
        "unique_tasks": len({row["task_id"] for row in rows}),
        "models": sorted({row["model"] for row in rows}),
        "modes": [item["label"] for item in rates],
        "pass_rates": rates,
    }


def generate(specs: list[BenchmarkSpec], out_dir: Path) -> list[Path]:
    written: list[Path] = []
    benchmark_files: dict[str, list[Path]] = {}
    for spec in specs:
        raw = read_jsonl(spec.log_path)
        rows = latest_rows(raw)
        summary = benchmark_summary(rows, raw)

        bench_dir = out_dir / spec.name
        benchmark_files[spec.name] = []
        charts = [
            ("pass_rate_by_mode_k.html", plot_pass_rate(rows, f"{spec.title}: hidden pass rate by mode/k")),
            ("outcome_counts_by_mode.html", plot_outcome_counts(rows, f"{spec.title}: pass/fail/timeout counts")),
            ("repair_outcomes.html", plot_repair_outcomes(rows, f"{spec.title}: repair outcome by mode/k")),
            ("self_test_signal.html", plot_self_test_signal(rows, f"{spec.title}: self-test signal vs hidden outcome")),
            ("tokens_vs_wall_time.html", plot_tokens_vs_wall(rows, f"{spec.title}: token spend vs wall time")),
            ("remaining_failures_heatmap.html", plot_remaining_failures(rows, f"{spec.title}: remaining failures by task")),
        ]
        for name, fig in charts:
            path = bench_dir / name
            write_html(fig, path)
            written.append(path)
            benchmark_files[spec.name].append(path)
        (bench_dir / "plotly_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        written.append(bench_dir / "plotly_summary.json")

    index_path = write_index(out_dir, benchmark_files)
    written.append(index_path)
    return written


def write_index(out_dir: Path, benchmark_files: dict[str, list[Path]]) -> Path:
    def rel(path: Path) -> str:
        return path.relative_to(out_dir).as_posix()

    sections = []
    for title, files in [
        (bench.replace("_", " ").title(), paths) for bench, paths in benchmark_files.items()
    ]:
        links = "\n".join(
            f'<li><a href="{rel(path)}">{path.stem.replace("_", " ")}</a></li>'
            for path in files
        )
        sections.append(f"<section><h2>{title}</h2><ul>{links}</ul></section>")

    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EvalPlus Plotly Charts</title>
  <style>
    body { font-family: Inter, Helvetica, Arial, sans-serif; margin: 32px; color: #111827; }
    h1 { font-size: 28px; margin-bottom: 8px; }
    p { color: #4b5563; max-width: 760px; }
    section { margin-top: 28px; }
    h2 { font-size: 18px; margin-bottom: 10px; }
    ul { line-height: 1.9; padding-left: 22px; }
    a { color: #1d4ed8; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>EvalPlus Plotly Charts</h1>
  <p>Interactive charts generated separately from the latest deduplicated rows in the HumanEval+ and MBPP+ probe logs.</p>
""" + "\n".join(sections) + """
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(html)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humaneval-log", default="results/humaneval_plus_probe_v1.jsonl")
    parser.add_argument("--mbpp-log", default="results/mbpp_plus_probe_v1.jsonl")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()

    specs = [
        BenchmarkSpec("humaneval_plus", "HumanEval+", Path(args.humaneval_log)),
        BenchmarkSpec("mbpp_plus", "MBPP+", Path(args.mbpp_log)),
    ]
    written = generate(specs, Path(args.out))
    print(f"wrote {len(written)} files to {args.out}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
