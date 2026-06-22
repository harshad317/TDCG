"""Benchmark external coding-agent CLIs on the same task/scoring format.

The agent workspace contains only prompt.md, public_tests.py, solution.py, and
agent guidance. hidden_tests.py is never copied into the agent workspace; hidden
scoring is performed afterwards by the harness.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .analysis import build_ablation_summary, format_ablation_summary
from .log import JsonlLogger, RunRecord, now
from .plot import make_all
from .sandbox import Sandbox, score_hidden


_NATURAL_PART_RE = re.compile(r"(\d+)")
AGENT_MODES = {
    "codex": "X_codex",
    "claude": "X_claude",
}
INITIAL_SOLUTION = "# Implement the task in this file.\n"
PROMPT_VERSION = "external_agents_v2_code_only"


@dataclass(frozen=True)
class WorkItem:
    index: int
    model: str
    seed: int
    task_dir: Path
    agent: str


@dataclass
class AgentCommand:
    argv: list[str]
    stdin: str | None = None


def natural_task_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    parts = _NATURAL_PART_RE.split(path.name)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in parts
        if part
    )


def parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"


def list_ollama_models() -> list[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            env=base_env(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"failed to list Ollama models: {e}") from e
    if proc.returncode != 0:
        raise SystemExit(f"ollama list failed: {proc.stderr.strip() or proc.stdout.strip()}")
    models: list[str] = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    if not models:
        raise SystemExit("ollama list returned no models")
    return models


def collect_tasks(args: argparse.Namespace) -> list[Path]:
    if args.benchmark:
        root = Path(args.bench_root) / args.benchmark
        if not root.exists():
            raise SystemExit(f"benchmark dir {root} not found")
        tasks = sorted((p for p in root.iterdir() if p.is_dir()), key=natural_task_key)
    elif args.tasks:
        tasks = [Path(t) for t in args.tasks]
    else:
        raise SystemExit("provide --benchmark or --tasks")
    return tasks[: args.limit] if args.limit is not None else tasks


def work_key(item: WorkItem) -> tuple[str, str, str, int]:
    return (item.model, item.task_dir.name, item.agent, item.seed)


def load_completed_work_keys(
    path: Path,
    *,
    batch_tag: str,
    models: set[str] | None = None,
    passed_only: bool = False,
) -> tuple[set[tuple[str, str, str, int]], int]:
    completed: set[tuple[str, str, str, int]] = set()
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
            row_model = str(row.get("model"))
            if models is not None and row_model not in models:
                continue
            extra = row.get("extra") or {}
            if extra.get("batch_tag") != batch_tag:
                continue
            agent = extra.get("external_agent")
            if agent not in AGENT_MODES:
                continue
            if passed_only and row.get("passed_hidden") is not True:
                continue
            try:
                completed.add((row_model, str(row["task_id"]), str(agent), int(row.get("seed", 0))))
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
    return completed, invalid_rows


def build_work_items(
    tasks: list[Path],
    agents: list[str],
    seeds: list[int],
    models: list[str],
) -> list[WorkItem]:
    items: list[WorkItem] = []
    index = 0
    for model in models:
        for seed in seeds:
            for task_dir in tasks:
                for agent in agents:
                    index += 1
                    items.append(WorkItem(index, model, seed, task_dir, agent))
    return items


def create_workspace(
    task_dir: Path,
    *,
    save_artifacts: bool,
    artifact_root: Path,
    batch_tag: str,
    model: str,
    agent: str,
    seed: int,
) -> tuple[Path, bool]:
    if save_artifacts:
        workspace = (
            artifact_root
            / batch_tag
            / model_slug(model)
            / agent
            / f"seed_{seed}"
            / task_dir.name
        )
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        workspace = Path(tempfile.mkdtemp(prefix=f"codehyp_{agent}_"))
        cleanup = True

    for name in ("prompt.md", "public_tests.py"):
        src = task_dir / name
        if src.exists():
            shutil.copy2(src, workspace / name)
    (workspace / "solution.py").write_text(INITIAL_SOLUTION, encoding="utf-8")
    guidance = external_agent_guidance(task_dir.name)
    (workspace / "AGENTS.md").write_text(guidance, encoding="utf-8")
    (workspace / "CLAUDE.md").write_text(guidance, encoding="utf-8")
    return workspace, cleanup


def external_agent_guidance(task_id: str) -> str:
    return f"""You are solving benchmark task {task_id}.

Rules:
- This is a non-interactive benchmark run; do not ask follow-up questions.
- Edit solution.py with the final Python implementation.
- You may read prompt.md and public_tests.py.
- You may run `python -m pytest -q public_tests.py`.
- Do not look for, create, infer from, or request hidden tests.
- Do not modify public_tests.py.
- Keep all work inside this directory.
- If file editing is unavailable, output exactly one fenced Python block with
  the complete final solution.py and no prose.
"""


def external_agent_prompt(task_dir: Path) -> str:
    task_id = task_dir.name
    prompt_text = (task_dir / "prompt.md").read_text(encoding="utf-8", errors="replace")
    public_tests = ""
    public_path = task_dir / "public_tests.py"
    if public_path.exists():
        public_tests = public_path.read_text(encoding="utf-8", errors="replace")
    return f"""You are running in non-interactive benchmark mode.

Implement the benchmark task {task_id}. Complete the task in this turn.

Required output behavior:
- Prefer editing solution.py with the final Python implementation.
- If you cannot edit files, print exactly one fenced Python block containing
  the complete final contents of solution.py.
- Do not print a plan.
- Do not ask questions.
- Do not include explanations, status text, or test commands in the fallback
  response. The fallback response must be code only.

Task prompt:

```markdown
{prompt_text}
```

Visible public tests:

```python
{public_tests}
```

Constraints:
- Write the final answer to solution.py.
- Do not modify public_tests.py.
- Do not search for hidden tests.
- You may run `python -m pytest -q public_tests.py`.
"""


def build_command(agent: str, model: str, prompt: str, args: argparse.Namespace) -> AgentCommand:
    if agent == "claude":
        return build_claude_command(model, prompt, args)
    if agent == "codex":
        return build_codex_command(model, prompt, args)
    raise ValueError(f"unknown agent: {agent}")


def build_claude_command(model: str, prompt: str, args: argparse.Namespace) -> AgentCommand:
    if args.claude_command == "direct":
        argv = [
            "claude",
            "-p",
            "--permission-mode",
            args.claude_permission_mode,
            "--max-turns",
            str(args.max_turns),
            "--model",
            model,
            prompt,
        ]
    else:
        argv = [
            "ollama",
            "launch",
            "claude",
            "--model",
            model,
            "--yes",
            "--",
            "-p",
            "--permission-mode",
            args.claude_permission_mode,
            "--max-turns",
            str(args.max_turns),
            prompt,
        ]
    return AgentCommand(argv=argv)


def build_codex_command(model: str, prompt: str, args: argparse.Namespace) -> AgentCommand:
    exec_args = [
        "exec",
        "--oss",
        "--local-provider",
        "ollama",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "--color",
        "never",
    ]
    if args.codex_command == "direct":
        argv = [
            "codex",
            "-a",
            "never",
            *exec_args,
            "-m",
            model,
            "-",
        ]
    else:
        argv = [
            "ollama",
            "launch",
            "codex",
            "--model",
            model,
            "--yes",
            "--",
            "-a",
            "never",
            *exec_args,
            "-",
        ]
    return AgentCommand(argv=argv, stdin=prompt)


def extract_python_fence(text: str) -> str | None:
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fence.strip() for fence in fences if "def " in fence or "class " in fence]
    if not candidates:
        return None
    return max(candidates, key=len).rstrip() + "\n"


def solution_needs_stdout_fallback(path: Path) -> bool:
    if not path.exists():
        return True
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = text.strip()
    if not stripped:
        return True
    return stripped == INITIAL_SOLUTION.strip()


def base_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "TERM": "dumb",
            "NO_COLOR": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONINTMAXSTRDIGITS": "0",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "ANTHROPIC_AUTH_TOKEN": env.get("ANTHROPIC_AUTH_TOKEN", "ollama"),
            "ANTHROPIC_BASE_URL": env.get("ANTHROPIC_BASE_URL", "http://localhost:11434"),
            "ANTHROPIC_API_KEY": env.get("ANTHROPIC_API_KEY", ""),
            "OPENAI_API_KEY": env.get("OPENAI_API_KEY", "ollama"),
        }
    )
    return env


def run_external_agent(
    item: WorkItem,
    args: argparse.Namespace,
    batch_tag: str,
) -> RunRecord:
    mode = AGENT_MODES[item.agent]
    record = RunRecord(
        task_id=item.task_dir.name,
        model=item.model,
        mode=mode,
        k=1,
        seed=item.seed,
    )
    record.extra["batch_tag"] = batch_tag
    record.extra["external_agent"] = item.agent
    record.extra["agent_model"] = item.model
    record.extra["agent_timeout_s"] = args.agent_timeout
    record.extra["hidden_timeout_s"] = args.hidden_timeout
    record.extra["score_policy"] = "hidden_after_agent"
    record.extra["stdout_code_fallback_enabled"] = args.stdout_code_fallback
    record.extra["prompt_version"] = PROMPT_VERSION

    workspace, cleanup = create_workspace(
        item.task_dir,
        save_artifacts=args.save_artifacts,
        artifact_root=Path(args.artifact_root),
        batch_tag=batch_tag,
        model=item.model,
        agent=item.agent,
        seed=item.seed,
    )
    record.extra["artifact_dir"] = str(workspace.resolve()) if args.save_artifacts else None

    prompt = external_agent_prompt(item.task_dir)
    (workspace / "agent_prompt.md").write_text(prompt, encoding="utf-8")
    command = build_command(item.agent, item.model, prompt, args)
    logged_argv = list(command.argv)
    if logged_argv and logged_argv[-1] == prompt:
        logged_argv[-1] = "<prompt>"
    record.extra["agent_command"] = logged_argv

    t0 = now()
    proc: subprocess.CompletedProcess[str] | None = None
    try:
        if args.dry_run:
            record.final_error_type = "dry_run"
            record.wall_time_s = round(now() - t0, 3)
            if cleanup:
                shutil.rmtree(workspace, ignore_errors=True)
            return record
        proc = subprocess.run(
            command.argv,
            input=command.stdin,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=args.agent_timeout,
            env=base_env(),
        )
        record.extra["agent_returncode"] = proc.returncode
        if args.save_artifacts:
            (workspace / "agent_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
            (workspace / "agent_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            record.final_error_type = f"agent_error:{item.agent}:returncode_{proc.returncode}"
    except subprocess.TimeoutExpired as e:
        record.extra["agent_timed_out"] = True
        record.final_error_type = f"agent_error:{item.agent}:TimeoutError:timed_out"
        if args.save_artifacts:
            stdout = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            (workspace / "agent_stdout.txt").write_text(stdout, encoding="utf-8")
            (workspace / "agent_stderr.txt").write_text(stderr, encoding="utf-8")
    except FileNotFoundError as e:
        record.final_error_type = f"agent_error:{item.agent}:missing_executable:{e.filename}"

    solution_path = workspace / "solution.py"
    if args.stdout_code_fallback and proc is not None and solution_needs_stdout_fallback(solution_path):
        fallback_code = extract_python_fence(proc.stdout or "")
        if fallback_code:
            solution_path.write_text(fallback_code, encoding="utf-8")
            record.extra["stdout_code_fallback_used"] = True
    if not solution_path.exists():
        record.final_error_type = record.final_error_type or "missing_solution"
        record.passed_public = False
        record.passed_hidden = False
        record.wall_time_s = round(now() - t0, 3)
        if cleanup:
            shutil.rmtree(workspace, ignore_errors=True)
        return record

    if solution_needs_stdout_fallback(solution_path):
        record.extra["agent_solution_unchanged"] = True
        if proc is not None:
            record.extra["agent_stdout_python_fence_present"] = extract_python_fence(proc.stdout or "") is not None
        record.passed_public = False
        record.passed_hidden = False
        record.final_error_type = record.final_error_type or "agent_no_edit"
        record.wall_time_s = round(now() - t0, 3)
        if cleanup:
            shutil.rmtree(workspace, ignore_errors=True)
        return record

    solution = solution_path.read_text(encoding="utf-8", errors="replace")
    sandbox = Sandbox(item.task_dir)
    try:
        sandbox.write("solution.py", solution)
        if (item.task_dir / "public_tests.py").exists():
            public = sandbox.run_pytest("public_tests.py", timeout=args.pytest_timeout)
            record.bash_calls_used += sandbox.bash_calls
            record.passed_public = public.returncode == 0
            record.extra["public_timed_out"] = public.timed_out
            if args.save_artifacts:
                (workspace / "public_stdout.txt").write_text(public.stdout, encoding="utf-8")
                (workspace / "public_stderr.txt").write_text(public.stderr, encoding="utf-8")
        hidden = score_hidden(item.task_dir, sandbox, timeout=args.hidden_timeout)
        record.passed_hidden = hidden.returncode == 0
        record.extra["hidden_timed_out"] = hidden.timed_out
        if args.save_artifacts:
            (workspace / "hidden_stdout.txt").write_text(hidden.stdout, encoding="utf-8")
            (workspace / "hidden_stderr.txt").write_text(hidden.stderr, encoding="utf-8")
        if record.passed_hidden:
            record.final_error_type = None
        elif hidden.timed_out:
            record.final_error_type = "hidden_timeout"
        elif record.final_error_type is None:
            record.final_error_type = "incorrect"
    finally:
        sandbox.cleanup()
        if cleanup:
            shutil.rmtree(workspace, ignore_errors=True)

    record.wall_time_s = round(now() - t0, 3)
    return record


def format_record_result(record: RunRecord) -> str:
    if record.passed_hidden is True:
        badge = "PASS"
    elif record.passed_hidden is False:
        badge = "FAIL"
    else:
        badge = "??"
    public = "PASS" if record.passed_public else ("FAIL" if record.passed_public is False else "-")
    hidden = "TIMEOUT" if record.extra.get("hidden_timed_out") else (
        "PASS" if record.passed_hidden else ("FAIL" if record.passed_hidden is False else "-")
    )
    return (
        f"{badge} public={public} hidden={hidden} "
        f"t={record.wall_time_s}s"
        + (f" error={record.final_error_type}" if record.final_error_type else "")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--models", default=None, help="comma-separated model ids; overrides --model")
    parser.add_argument(
        "--models-from-ollama",
        action="store_true",
        help="benchmark every model reported by `ollama list`; overrides --model/--models",
    )
    parser.add_argument("--agents", default="codex,claude", help="comma-separated: codex,claude")
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--bench-root", default="tasks_bench")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--pytest-timeout", type=int, default=10)
    parser.add_argument("--hidden-timeout", type=int, default=600)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--log", default="results/external_agents.jsonl")
    parser.add_argument("--batch-tag", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--artifact-root", default="results/external_artifacts")
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-stdout-code-fallback",
        dest="stdout_code_fallback",
        action="store_false",
        help="disable parsing a final fenced Python block when the agent fails to edit solution.py",
    )
    parser.set_defaults(stdout_code_fallback=True)
    parser.add_argument("--codex-command", choices=["ollama-launch", "direct"], default="direct")
    parser.add_argument("--claude-command", choices=["ollama-launch", "direct"], default="ollama-launch")
    parser.add_argument(
        "--claude-permission-mode",
        default="bypassPermissions",
        help="passed to Claude Code --permission-mode; run only in isolated workspaces",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    agents = parse_csv_str(args.agents)
    unknown_agents = [agent for agent in agents if agent not in AGENT_MODES]
    if unknown_agents:
        raise SystemExit(f"unknown agents: {', '.join(unknown_agents)}")
    if args.models_from_ollama:
        models = list_ollama_models()
    elif args.models:
        models = parse_csv_str(args.models)
    else:
        models = [args.model]
    models = dedupe_preserve_order(models)
    if not models:
        raise SystemExit("no models selected")
    seeds = parse_csv_ints(args.seeds) if args.seeds else [args.seed]
    tasks = collect_tasks(args)
    work_items = build_work_items(tasks, agents, seeds, models)

    log_path = Path(args.log)
    if args.resume or args.rerun_failed:
        completed, invalid_rows = load_completed_work_keys(
            log_path,
            batch_tag=args.batch_tag,
            models=set(models),
            passed_only=args.rerun_failed,
        )
        before = len(work_items)
        work_items = [item for item in work_items if work_key(item) not in completed]
        suffix = " (passed rows only)" if args.rerun_failed else ""
        print(f"resume: skipped {before - len(work_items)}/{before} completed cases from {log_path}{suffix}")
        if invalid_rows:
            print(f"resume: ignored {invalid_rows} invalid rows", file=sys.stderr)

    logger = JsonlLogger(log_path)
    print(
        f"running {len(work_items)} external-agent cases across "
        f"{len(models)} model(s), {len(agents)} agent(s), {len(seeds)} seed(s) "
        f"with {args.jobs} workers"
    )

    rows_for_summary: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_external_agent, item, args, args.batch_tag): item
            for item in work_items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as e:  # keep long benchmark runs moving
                record = RunRecord(
                    task_id=item.task_dir.name,
                    model=item.model,
                    mode=AGENT_MODES[item.agent],
                    k=1,
                    seed=item.seed,
                    passed_hidden=False,
                    final_error_type=f"harness_error:{type(e).__name__}:{e}",
                    extra={"batch_tag": args.batch_tag, "external_agent": item.agent},
                )
            logger.write(record)
            rows_for_summary.append(asdict(record))
            print(
                f"[{item.index}] model={item.model} seed={item.seed} task={item.task_dir.name} "
                f"agent={item.agent} ... {format_record_result(record)}",
                flush=True,
            )

    if rows_for_summary:
        summary = build_ablation_summary(rows_for_summary)
        print("\n" + format_ablation_summary(summary))

    if not args.no_plot and not args.dry_run:
        out_dir = Path(args.plot_dir) if args.plot_dir else Path("results/plots") / args.batch_tag
        plot_model = models[0] if len(models) == 1 else None
        plots = make_all(log_path, out_dir, model=plot_model, batch_tag=args.batch_tag)
        print(f"wrote {len(plots)} plots to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
