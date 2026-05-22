"""Agent: drives the model across modes A/B/C/D/E.

Mode capabilities (from final-final plan):
  A: code only, no tests, no execution                — one-shot baseline
  B: code + model-written tests, no execution         — test-thinking alone
  C: code, model can run public_tests.py              — feedback w/ reliable tests
  D: code + model-written tests, model can run them   — full self-loop
  E: code + model tests + public tests, runs both     — practical ceiling
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .log import RunRecord, now
from .models import ChatResult, OllamaClient
from .sandbox import RunResult, Sandbox, score_hidden


MODES = {"A", "B", "C", "D", "E"}
MAX_BASH_CALLS = 10

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class AgentConfig:
    mode: str
    k: int
    max_bash_calls: int = MAX_BASH_CALLS
    pytest_timeout: int = 10
    hidden_timeout: int = 60
    score_hidden_each_iter: bool = False


@dataclass
class StepLog:
    iteration: int
    test_output: str = ""
    public_passed: Optional[bool] = None
    self_passed: Optional[bool] = None


# ----------------------------------------------------------------------------- prompts


SYSTEM_BASE = (
    "You are a careful Python programmer. When you produce code, put each file in "
    "a separate fenced block whose first line is a comment naming the file, like:\n"
    "```python\n# solution.py\ndef foo(): ...\n```\n"
    "Always include the full file contents — do not output diffs or partials. "
    "When asked for multiple files, output all required files every time."
)


def initial_user_prompt(prompt_md: str, starter: str, mode: str, public_tests: Optional[str]) -> str:
    parts = [
        "Task description:",
        prompt_md.strip(),
        "",
        "Current solution.py:",
        "```python",
        starter.strip(),
        "```",
    ]
    if mode in ("C", "E") and public_tests is not None:
        parts += [
            "",
            "public_tests.py (you may run these for feedback):",
            "```python",
            public_tests.strip(),
            "```",
        ]
    if mode == "A":
        parts += ["", "Produce the final solution.py. Output only the file, in one fenced block."]
    elif mode == "B":
        parts += [
            "",
            "Produce both:",
            "  1. solution.py — your implementation",
            "  2. self_tests.py — additional unit tests you would run if you could",
            "Output two fenced blocks in that order. You will NOT be able to run tests.",
        ]
    elif mode == "C":
        parts += [
            "",
            "Produce solution.py. After each version, the harness will run public_tests.py "
            "and feed you the result. Iterate until tests pass or budget runs out.",
        ]
    elif mode == "D":
        parts += [
            "",
            "Protocol:",
            "  1. First write your best final solution.py for the task.",
            "  2. Then write self_tests.py to check whether that solution satisfies the input prompt.",
            "  3. The harness will run self_tests.py with pytest and feed you the terminal result.",
            "  4. If self_tests.py fails, repair solution.py and/or self_tests.py, then output both full files again.",
            "  5. Stop only when your self_tests.py passes or the iteration budget runs out.",
            "",
            "Output exactly two fenced Python blocks every time, in this order:",
            "```python",
            "# solution.py",
            "<full implementation>",
            "```",
            "```python",
            "# self_tests.py",
            "<pytest tests importing from solution.py>",
            "```",
            "",
            "self_tests.py must import the public function/class from solution.py and contain pytest-compatible tests.",
            "Do not omit self_tests.py. If you are uncertain, still write at least one simple pytest test derived from the examples or docstring.",
            "Do not use prose outside the two fenced code blocks.",
            "Note: public_tests.py is NOT provided in this mode.",
        ]
    elif mode == "E":
        parts += [
            "",
            "Produce both solution.py and self_tests.py. The harness will run "
            "public_tests.py AND self_tests.py and feed you both results. Iterate.",
        ]
    return "\n".join(parts)


def feedback_prompt(public_res: Optional[RunResult], self_res: Optional[RunResult]) -> str:
    chunks = []
    if public_res is not None:
        chunks.append(_format_pytest_result("public_tests.py", public_res))
    if self_res is not None:
        chunks.append(_format_pytest_result("self_tests.py", self_res))
    chunks.append(
        "Update the files and output the full file contents again. "
        "For self-test modes, output both solution.py first and self_tests.py second."
    )
    return "\n\n".join(chunks)


def _format_pytest_result(label: str, res: RunResult) -> str:
    status = "PASSED" if res.returncode == 0 else f"FAILED (exit {res.returncode})"
    tail = (res.stdout + "\n" + res.stderr).strip()
    if len(tail) > 2000:
        tail = tail[-2000:]
    return f"[{label}] {status}\n```\n{tail}\n```"


def _visible_passed(mode: str, public_res: Optional[RunResult], self_res: Optional[RunResult]) -> bool:
    required: list[RunResult | None] = []
    if mode in ("C", "E"):
        required.append(public_res)
    if mode in ("D", "E"):
        required.append(self_res)
    return bool(required) and all(res is not None and res.returncode == 0 for res in required)


def _result_passed(res: Optional[RunResult]) -> Optional[bool]:
    return (res.returncode == 0) if res is not None else None


def _trace_hidden_result(
    iteration: int,
    mode: str,
    public_res: Optional[RunResult],
    self_res: Optional[RunResult],
    hidden_res: RunResult,
) -> dict:
    return {
        "iteration": iteration,
        "public_passed": _result_passed(public_res),
        "self_passed": _result_passed(self_res),
        "visible_passed": _visible_passed(mode, public_res, self_res) if mode in ("C", "D", "E") else None,
        "hidden_passed": hidden_res.returncode == 0,
        "hidden_timed_out": hidden_res.timed_out,
        "hidden_returncode": hidden_res.returncode,
    }


def _add_repair_metrics(record: RunRecord, iteration_trace: list[dict]) -> None:
    if not iteration_trace:
        record.extra["initial_hidden_pass"] = None
        record.extra["hidden_improved"] = None
        record.extra["fixed_by_self_tests"] = None
        return

    initial_hidden_pass = iteration_trace[0]["hidden_passed"]
    final_hidden_pass = record.passed_hidden
    self_failed_before_final = any(
        step.get("self_passed") is False for step in iteration_trace[:-1]
    )
    hidden_improved = initial_hidden_pass is False and final_hidden_pass is True
    fixed_by_self_tests = (
        record.mode in ("D", "E")
        and hidden_improved
        and self_failed_before_final
    )

    record.extra["initial_hidden_pass"] = initial_hidden_pass
    record.extra["hidden_improved"] = hidden_improved
    record.extra["self_failed_before_final"] = self_failed_before_final
    record.extra["fixed_by_self_tests"] = fixed_by_self_tests


# ----------------------------------------------------------------------------- code extraction


def _canonical_generated_name(name: str) -> str:
    base = Path(name).name
    if base in {"tests.py", "test.py", "test_solution.py", "solution_tests.py"}:
        return "self_tests.py"
    return base


def _split_marked_block(block: str) -> list[tuple[str, str]]:
    """Split one fenced block containing `# solution.py` / `# self_tests.py` markers."""
    marker_re = re.compile(
        r"^#\s*([\w.\-/]*(?:solution|self_tests|tests?|test_solution|solution_tests)\.py)\b.*$",
        re.MULTILINE,
    )
    markers = list(marker_re.finditer(block))
    pieces: list[tuple[str, str]] = []
    for i, marker in enumerate(markers):
        start = block.find("\n", marker.start())
        start = len(block) if start == -1 else start + 1
        end = markers[i + 1].start() if i + 1 < len(markers) else len(block)
        name = _canonical_generated_name(marker.group(1))
        body = block[start:end].strip("\n")
        if body.strip():
            pieces.append((name, body + "\n"))
    return pieces


def _looks_like_pytest_block(block: str) -> bool:
    return bool(re.search(r"(^|\n)\s*def\s+test_", block)) or "from solution import" in block


def extract_files(text: str) -> dict[str, str]:
    """Pull out generated files from fenced Python blocks.

    Preferred format is a first-line marker like `# solution.py`, but small
    models often emit `# tests.py`, omit headers on the second block, or put both
    files in one fenced block. We accept those forms to keep Mode D focused on
    self-testing ability rather than brittle formatting.
    """
    out: dict[str, str] = {}
    unnamed: list[str] = []
    for block in CODE_FENCE_RE.findall(text):
        block = block.strip("\n")
        first_line = block.splitlines()[0] if block else ""
        marked = _split_marked_block(block)
        if marked:
            for name, body in marked:
                out[name] = body.lstrip("\n")
            continue
        m = re.match(r"#\s*([\w.\-/]+\.py)\b", first_line)
        if m:
            name = _canonical_generated_name(m.group(1))
            body = "\n".join(block.splitlines()[1:])
            out[name] = body.lstrip("\n")
        else:
            unnamed.append(block)
    for block in unnamed:
        if "solution.py" not in out and not _looks_like_pytest_block(block):
            out["solution.py"] = block
        elif "self_tests.py" not in out:
            out["self_tests.py"] = block
        elif "solution.py" not in out:
            out["solution.py"] = block
    if "solution.py" not in out and unnamed:
        out["solution.py"] = unnamed[0]
    return out


# ----------------------------------------------------------------------------- agent loop


def run_task(
    client: OllamaClient,
    task_dir: Path,
    cfg: AgentConfig,
    record: RunRecord,
) -> RunRecord:
    if cfg.mode not in MODES:
        raise ValueError(f"unknown mode: {cfg.mode}")
    if cfg.mode in ("A", "B") and cfg.k != 1:
        raise ValueError(f"mode {cfg.mode} requires k=1")

    sandbox = Sandbox(task_dir)
    t0 = now()
    try:
        prompt_md = (task_dir / "prompt.md").read_text()
        starter = sandbox.read("solution.py")
        public_tests_text = sandbox.read("public_tests.py") if cfg.mode in ("C", "E") else None

        messages = [
            {"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": initial_user_prompt(prompt_md, starter, cfg.mode, public_tests_text)},
        ]

        last_public: Optional[RunResult] = None
        last_self: Optional[RunResult] = None
        iters = 0
        self_tests_present = False
        ran_public_tests = False
        ran_self_tests = False
        iteration_trace: list[dict] = []
        last_hidden_probe: Optional[RunResult] = None

        for step in range(cfg.k):
            iters = step + 1
            try:
                reply = client.chat(messages)
            except Exception as e:
                record.final_error_type = f"model_error:{type(e).__name__}"
                break
            record.tokens_in += reply.tokens_in
            record.tokens_out += reply.tokens_out
            messages.append({"role": "assistant", "content": reply.text})

            files = extract_files(reply.text)
            if "solution.py" in files:
                sandbox.write("solution.py", files["solution.py"])
            if cfg.mode in ("B", "D", "E") and "self_tests.py" in files:
                sandbox.write("self_tests.py", files["self_tests.py"])
                self_tests_present = True

            public_res: Optional[RunResult] = None
            self_res: Optional[RunResult] = None
            if cfg.mode in ("C", "E"):
                if sandbox.bash_calls < cfg.max_bash_calls:
                    public_res = sandbox.run_pytest("public_tests.py", cfg.pytest_timeout)
                    last_public = public_res
                    ran_public_tests = True
                else:
                    public_res = RunResult(124, "", "bash call budget exhausted before public_tests.py", False)
                    last_public = public_res
            if cfg.mode in ("D", "E"):
                has_self = (sandbox.tmp / "self_tests.py").exists()
                if has_self and sandbox.bash_calls < cfg.max_bash_calls:
                    self_res = sandbox.run_pytest("self_tests.py", cfg.pytest_timeout)
                    last_self = self_res
                    self_tests_present = True
                    ran_self_tests = True
                elif not has_self:
                    self_res = RunResult(2, "", "self_tests.py was not provided by the model.", False)
                    last_self = self_res
                else:
                    self_res = RunResult(124, "", "bash call budget exhausted before self_tests.py", False)
                    last_self = self_res

            if cfg.score_hidden_each_iter:
                last_hidden_probe = score_hidden(task_dir, sandbox, timeout=cfg.hidden_timeout)
                iteration_trace.append(
                    _trace_hidden_result(iters, cfg.mode, public_res, self_res, last_hidden_probe)
                )

            # Non-exec modes: stop after first generation.
            if cfg.mode in ("A", "B"):
                break

            visible_pass = _visible_passed(cfg.mode, public_res, self_res)
            if visible_pass:
                break
            if step == cfg.k - 1:
                break  # no point asking again, we won't run another iter

            messages.append({"role": "user", "content": feedback_prompt(public_res, self_res)})

        # Score against hidden tests.
        if cfg.score_hidden_each_iter and last_hidden_probe is not None:
            hidden = last_hidden_probe
        else:
            hidden = score_hidden(task_dir, sandbox, timeout=cfg.hidden_timeout)
        record.iterations_used = iters
        record.bash_calls_used = sandbox.bash_calls
        record.extra["pytest_timeout_s"] = cfg.pytest_timeout
        record.extra["hidden_timeout_s"] = cfg.hidden_timeout
        record.extra["hidden_timed_out"] = hidden.timed_out
        record.extra["score_hidden_each_iter"] = cfg.score_hidden_each_iter
        if cfg.score_hidden_each_iter:
            record.extra["iteration_trace"] = iteration_trace
        record.extra["self_tests_present"] = self_tests_present if cfg.mode in ("B", "D", "E") else None
        record.extra["ran_public_tests"] = ran_public_tests if cfg.mode in ("C", "E") else None
        record.extra["ran_self_tests"] = ran_self_tests if cfg.mode in ("D", "E") else None
        record.passed_public = (last_public.returncode == 0) if last_public is not None else None
        record.passed_self = (last_self.returncode == 0) if last_self is not None else None
        record.passed_hidden = hidden.returncode == 0
        if not record.passed_hidden:
            visible_passed = _visible_passed(cfg.mode, last_public, last_self)
            if cfg.mode in ("D", "E") and not self_tests_present:
                record.final_error_type = "missing_self_tests"
            elif hidden.returncode == 124:
                record.final_error_type = "overfit_hidden_timeout" if visible_passed else "hidden_timeout"
            elif visible_passed:
                if cfg.mode == "C":
                    record.final_error_type = "overfit_public"
                elif cfg.mode == "D":
                    record.final_error_type = "overfit_self"
                else:
                    record.final_error_type = "overfit_public_self"
            else:
                record.final_error_type = "incorrect"
        _add_repair_metrics(record, iteration_trace)
    finally:
        record.wall_time_s = round(now() - t0, 3)
        sandbox.cleanup()
    return record
