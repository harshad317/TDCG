"""Agent: drives the model across modes A/B/C/D/D_sep/D_dual/D_val/E.

Mode capabilities (from final-final plan):
  A: code only, no tests, no execution                — one-shot baseline
  B: code + model-written tests, no execution         — test-thinking alone
  C: code, model can run public_tests.py              — feedback w/ reliable tests
  D: code + model-written tests, model can run them   — full self-loop
  D_sep: write code, then tests in a separate call     — cleaner self-loop ablation
  D_dual: code model + test model with skill files     — two-model self-loop
  D_val: code model + test model + test validator      — validated self-loop
  E: code + model tests + public tests, runs both     — practical ceiling
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .log import RunRecord, now
from .models import ChatResult, OllamaClient
from .sandbox import RunResult, Sandbox, score_hidden, score_self_tests_on_reference


MODES = {"A", "B", "C", "D", "D_sep", "D_dual", "D_val", "E"}
MAX_BASH_CALLS = 10
SELF_TEST_MODES = {"B", "D", "D_sep", "D_dual", "D_val", "E"}
EXEC_SELF_TEST_MODES = {"D", "D_sep", "D_dual", "D_val", "E"}
DEFAULT_CODE_SKILL_PATH = Path("agent_skills/code_writer/skills.md")
DEFAULT_TEST_SKILL_PATH = Path("agent_skills/test_writer/skills.md")
DEFAULT_VALIDATOR_SKILL_PATH = Path("agent_skills/test_validator/skills.md")
MAX_TEST_VALIDATION_ATTEMPTS = 2

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class AgentConfig:
    mode: str
    k: int
    max_bash_calls: int = MAX_BASH_CALLS
    pytest_timeout: int = 10
    hidden_timeout: int = 60
    score_hidden_each_iter: bool = False
    code_skill_path: Path = DEFAULT_CODE_SKILL_PATH
    test_skill_path: Path = DEFAULT_TEST_SKILL_PATH
    validator_skill_path: Path = DEFAULT_VALIDATOR_SKILL_PATH
    artifact_root: Optional[Path] = None


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


def _read_skill(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError as e:
        raise RuntimeError(f"could not read skill file {path}: {e}") from e


def _system_with_skill(skill_text: str) -> str:
    return "\n\n".join([SYSTEM_BASE, "Skill instructions:", skill_text.strip()])


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
    elif mode == "D_sep":
        parts += [
            "",
            "Separated self-test protocol, step 1:",
            "Write your best final solution.py for the task.",
            "Do not write tests yet. The harness will ask for self_tests.py in a separate model call.",
            "",
            "Output exactly one fenced Python block:",
            "```python",
            "# solution.py",
            "<full implementation>",
            "```",
            "",
            "Do not use prose outside the fenced code block.",
            "Note: public_tests.py is NOT provided in this mode.",
        ]
    elif mode == "E":
        parts += [
            "",
            "Produce both solution.py and self_tests.py. The harness will run "
            "public_tests.py AND self_tests.py and feed you both results. Iterate.",
        ]
    return "\n".join(parts)


def dual_code_initial_prompt(prompt_md: str, starter: str) -> str:
    return "\n".join(
        [
            "Two-model protocol, code-writer step:",
            "Write the best final solution.py for the task. Do not write tests.",
            "",
            "Task description:",
            prompt_md.strip(),
            "",
            "Current solution.py:",
            "```python",
            starter.strip(),
            "```",
        ]
    )


def dual_test_user_prompt(prompt_md: str, solution: str) -> str:
    return "\n".join(
        [
            "Two-model protocol, test-writer step:",
            "Write self_tests.py to check whether the current solution.py satisfies the task prompt.",
            "",
            "Task description:",
            prompt_md.strip(),
            "",
            "Current solution.py:",
            "```python",
            solution.strip(),
            "```",
        ]
    )


def test_validation_user_prompt(prompt_md: str, solution: str, self_tests: str) -> str:
    return "\n".join(
        [
            "Validated self-test protocol, validator step:",
            "Decide whether self_tests.py is a valid test oracle for the task prompt.",
            "Audit every assertion. Manually derive each expected value from the task prompt.",
            "Reject the entire file if any assertion has a wrong expected value or tests behavior outside the prompt.",
            "",
            "Task description:",
            prompt_md.strip(),
            "",
            "Current solution.py:",
            "```python",
            solution.strip(),
            "```",
            "",
            "Candidate self_tests.py:",
            "```python",
            self_tests.strip(),
            "```",
        ]
    )


def self_test_revision_prompt(
    prompt_md: str,
    solution: str,
    self_tests: str,
    validation_reason: str,
) -> str:
    return "\n\n".join(
        [
            "The validator rejected self_tests.py.",
            "Rewrite self_tests.py so it is a valid pytest oracle for the task prompt.",
            "Do not change solution.py.",
            "Validator feedback:",
            validation_reason.strip() or "No validator reason was provided.",
            "Task description:",
            prompt_md.strip(),
            "Current solution.py:",
            "```python\n" + solution.strip() + "\n```",
            "Rejected self_tests.py:",
            "```python\n" + self_tests.strip() + "\n```",
            "Output exactly one fenced Python block whose first line is `# self_tests.py`.",
            "Do not include prose outside the fenced code block.",
        ]
    )


def dual_code_repair_prompt(
    prompt_md: str,
    solution: str,
    self_res: RunResult,
    self_tests: str,
) -> str:
    return "\n\n".join(
        [
            "Two-model protocol, code repair step:",
            "Task description:",
            prompt_md.strip(),
            "Current solution.py:",
            "```python\n" + solution.strip() + "\n```",
            "Frozen self_tests.py:",
            "```python\n" + self_tests.strip() + "\n```",
            _format_pytest_result("self_tests.py", self_res),
            "Debug the failure above. Compare the actual value against the expected value, "
            "identify the smallest algorithmic mistake, and update solution.py only.",
            "The repaired solution must satisfy the task prompt, not merely hard-code these tests.",
            "Do not modify or rewrite self_tests.py.",
            "Output the full solution.py file in one fenced Python block with `# solution.py` as the first line.",
        ]
    )


def validated_code_repair_prompt(
    prompt_md: str,
    solution: str,
    self_res: RunResult,
    self_tests: str,
) -> str:
    return "\n\n".join(
        [
            "Validated self-test protocol, code repair step:",
            "Task description:",
            prompt_md.strip(),
            "Current solution.py:",
            "```python\n" + solution.strip() + "\n```",
            "Frozen validator-approved self_tests.py:",
            "```python\n" + self_tests.strip() + "\n```",
            _format_pytest_result("self_tests.py", self_res),
            "Debug the failure above. Compare the actual value against the expected value, "
            "identify the smallest algorithmic mistake, and update solution.py only.",
            "The repaired solution must satisfy the task prompt, not merely hard-code these tests.",
            "Do not modify or rewrite self_tests.py.",
            "Output the full solution.py file in one fenced Python block with `# solution.py` as the first line.",
        ]
    )


def self_tests_user_prompt(prompt_md: str, solution: str) -> str:
    return "\n".join(
        [
            "Separated self-test protocol, step 2:",
            "Now write self_tests.py to test whether the current solution.py satisfies the task prompt.",
            "",
            "Task description:",
            prompt_md.strip(),
            "",
            "Current solution.py:",
            "```python",
            solution.strip(),
            "```",
            "",
            "Output exactly one fenced Python block:",
            "```python",
            "# self_tests.py",
            "<pytest tests importing from solution.py>",
            "```",
            "",
            "self_tests.py must import the public function/class from solution.py and contain pytest-compatible tests.",
            "Do not use prose outside the fenced code block.",
        ]
    )


def separated_solution_repair_prompt(
    prompt_md: str,
    solution: str,
    self_tests: str,
    self_res: RunResult,
) -> str:
    return "\n\n".join(
        [
            "Separated self-test protocol, repair step:",
            "Task description:",
            prompt_md.strip(),
            "Current solution.py:",
            "```python\n" + solution.strip() + "\n```",
            "Frozen self_tests.py:",
            "```python\n" + self_tests.strip() + "\n```",
            _format_pytest_result("self_tests.py", self_res),
            "Debug the failure above. Compare the actual value against the expected value, "
            "identify the smallest algorithmic mistake, and update solution.py only.",
            "The repaired solution must satisfy the task prompt, not merely hard-code these tests.",
            "Do not write or modify tests in this response.",
            "Output the full solution.py file in one fenced Python block with `# solution.py` as the first line.",
        ]
    )


def feedback_prompt(
    public_res: Optional[RunResult],
    self_res: Optional[RunResult],
    solution: str,
    self_tests: Optional[str],
) -> str:
    chunks = [
        "Current solution.py:",
        "```python\n" + solution.strip() + "\n```",
    ]
    if self_tests is not None:
        chunks += [
            "Current self_tests.py:",
            "```python\n" + self_tests.strip() + "\n```",
        ]
    if public_res is not None:
        chunks.append(_format_pytest_result("public_tests.py", public_res))
    if self_res is not None:
        chunks.append(_format_pytest_result("self_tests.py", self_res))
    chunks.append(
        "Repair using the terminal feedback above. Compare actual vs expected output, "
        "fix the underlying algorithm, and output the full file contents again. "
        "For self-test modes, output both solution.py first and self_tests.py second. "
        "Only change self_tests.py if the tests contradict the task prompt or are invalid Python."
    )
    return "\n\n".join(chunks)


def _format_pytest_result(label: str, res: RunResult) -> str:
    status = "PASSED" if res.returncode == 0 else f"FAILED (exit {res.returncode})"
    tail = (res.stdout + "\n" + res.stderr).strip()
    if len(tail) > 2000:
        tail = tail[-2000:]
    return f"[{label}] {status}\n```\n{tail}\n```"


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "unknown"


def _save_result_artifact(path: Path, res: Optional[RunResult]) -> None:
    if res is None:
        return
    path.write_text(
        "\n".join(
            [
                f"returncode={res.returncode}",
                f"timed_out={res.timed_out}",
                "",
                "STDOUT:",
                res.stdout,
                "",
                "STDERR:",
                res.stderr,
            ]
        )
    )


def _save_iteration_artifacts(
    cfg: AgentConfig,
    record: RunRecord,
    sandbox: Sandbox,
    iteration: int,
    public_res: Optional[RunResult],
    self_res: Optional[RunResult],
    hidden_res: Optional[RunResult],
) -> None:
    if cfg.artifact_root is None:
        return

    run_dir = (
        Path(cfg.artifact_root)
        / f"seed_{record.seed}"
        / _safe_path_part(record.task_id)
        / f"{_safe_path_part(record.mode)}_k{record.k}"
    )
    iter_dir = run_dir / f"iter_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    record.extra["artifact_dir"] = str(run_dir.resolve())

    for name in ("solution.py", "self_tests.py", "public_tests.py"):
        src = sandbox.tmp / name
        if src.exists():
            (iter_dir / name).write_text(src.read_text())
    _save_result_artifact(iter_dir / "public_result.txt", public_res)
    _save_result_artifact(iter_dir / "self_result.txt", self_res)
    _save_result_artifact(iter_dir / "hidden_result.txt", hidden_res)


def _visible_passed(mode: str, public_res: Optional[RunResult], self_res: Optional[RunResult]) -> bool:
    required: list[RunResult | None] = []
    if mode in ("C", "E"):
        required.append(public_res)
    if mode in EXEC_SELF_TEST_MODES:
        required.append(self_res)
    return bool(required) and all(res is not None and res.returncode == 0 for res in required)


def _result_passed(res: Optional[RunResult]) -> Optional[bool]:
    return (res.returncode == 0) if res is not None else None


def _model_error(e: Exception, role: str = "model") -> str:
    message = str(e).replace("\n", " ").strip()
    if len(message) > 160:
        message = message[:157] + "..."
    return f"model_error:{role}:{type(e).__name__}" + (f":{message}" if message else "")


def _parse_test_validation(text: str) -> tuple[bool, str]:
    match = re.search(r"(?im)^\s*TESTS_VALID\s*:\s*(yes|no|true|false)\s*$", text)
    valid = bool(match and match.group(1).lower() in {"yes", "true"})
    reason_match = re.search(r"(?ims)^\s*REASON\s*:\s*(.*)$", text)
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    if len(reason) > 1200:
        reason = reason[:1197] + "..."
    return valid, reason


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
        "visible_passed": (
            _visible_passed(mode, public_res, self_res)
            if mode in ("C", "D", "D_sep", "D_dual", "D_val", "E")
            else None
        ),
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
        record.mode in EXEC_SELF_TEST_MODES
        and hidden_improved
        and self_failed_before_final
    )

    record.extra["initial_hidden_pass"] = initial_hidden_pass
    record.extra["hidden_improved"] = hidden_improved
    record.extra["self_failed_before_final"] = self_failed_before_final
    record.extra["fixed_by_self_tests"] = fixed_by_self_tests


def _score_and_finalize(
    task_dir: Path,
    sandbox: Sandbox,
    cfg: AgentConfig,
    record: RunRecord,
    iters: int,
    last_public: Optional[RunResult],
    last_self: Optional[RunResult],
    self_tests_present: bool,
    ran_public_tests: bool,
    ran_self_tests: bool,
    iteration_trace: list[dict],
    last_hidden_probe: Optional[RunResult],
) -> None:
    if (
        record.final_error_type
        and record.final_error_type.startswith("model_error:")
        and record.tokens_out == 0
    ):
        record.iterations_used = iters
        record.bash_calls_used = sandbox.bash_calls
        record.extra["pytest_timeout_s"] = cfg.pytest_timeout
        record.extra["hidden_timeout_s"] = cfg.hidden_timeout
        record.extra["hidden_timed_out"] = False
        record.extra["score_hidden_each_iter"] = cfg.score_hidden_each_iter
        if cfg.score_hidden_each_iter:
            record.extra["iteration_trace"] = iteration_trace
        record.extra["self_tests_present"] = self_tests_present if cfg.mode in SELF_TEST_MODES else None
        record.extra["ran_public_tests"] = ran_public_tests if cfg.mode in ("C", "E") else None
        record.extra["ran_self_tests"] = ran_self_tests if cfg.mode in EXEC_SELF_TEST_MODES else None
        record.passed_public = (last_public.returncode == 0) if last_public is not None else None
        record.passed_self = (last_self.returncode == 0) if last_self is not None else None
        record.passed_hidden = False
        _add_repair_metrics(record, iteration_trace)
        return

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
    record.extra["self_tests_present"] = self_tests_present if cfg.mode in SELF_TEST_MODES else None
    record.extra["ran_public_tests"] = ran_public_tests if cfg.mode in ("C", "E") else None
    record.extra["ran_self_tests"] = ran_self_tests if cfg.mode in EXEC_SELF_TEST_MODES else None
    record.passed_public = (last_public.returncode == 0) if last_public is not None else None
    record.passed_self = (last_self.returncode == 0) if last_self is not None else None
    record.passed_hidden = hidden.returncode == 0
    if not record.passed_hidden and not record.final_error_type:
        visible_passed = _visible_passed(cfg.mode, last_public, last_self)
        if cfg.mode in EXEC_SELF_TEST_MODES and not self_tests_present:
            record.final_error_type = "missing_self_tests"
        elif hidden.returncode == 124:
            record.final_error_type = "overfit_hidden_timeout" if visible_passed else "hidden_timeout"
        elif visible_passed:
            if cfg.mode == "C":
                record.final_error_type = "overfit_public"
            elif cfg.mode in ("D", "D_sep", "D_dual", "D_val"):
                record.final_error_type = "overfit_self"
            else:
                record.final_error_type = "overfit_public_self"
        else:
            record.final_error_type = "incorrect"
    _add_repair_metrics(record, iteration_trace)


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


def _run_task_d_sep(
    client: OllamaClient,
    task_dir: Path,
    cfg: AgentConfig,
    record: RunRecord,
) -> RunRecord:
    """Mode D_sep: write solution, write tests separately, then repair solution only."""
    sandbox = Sandbox(task_dir)
    t0 = now()
    try:
        prompt_md = (task_dir / "prompt.md").read_text()
        starter = sandbox.read("solution.py")

        messages = [
            {"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": initial_user_prompt(prompt_md, starter, cfg.mode, None)},
        ]

        last_public: Optional[RunResult] = None
        last_self: Optional[RunResult] = None
        iters = 0
        self_tests_present = False
        ran_public_tests = False
        ran_self_tests = False
        iteration_trace: list[dict] = []
        last_hidden_probe: Optional[RunResult] = None
        self_test_generation_attempts = 0
        record.extra["self_tests_separate_call"] = True
        record.extra["self_tests_frozen_after_generation"] = True

        try:
            solution_reply = client.chat(messages)
        except Exception as e:
            record.final_error_type = _model_error(e)
            _score_and_finalize(
                task_dir,
                sandbox,
                cfg,
                record,
                iters,
                last_public,
                last_self,
                self_tests_present,
                ran_public_tests,
                ran_self_tests,
                iteration_trace,
                last_hidden_probe,
            )
            return record
        record.tokens_in += solution_reply.tokens_in
        record.tokens_out += solution_reply.tokens_out
        messages.append({"role": "assistant", "content": solution_reply.text})

        files = extract_files(solution_reply.text)
        if "solution.py" in files:
            sandbox.write("solution.py", files["solution.py"])

        solution = sandbox.read("solution.py")
        messages.append({"role": "user", "content": self_tests_user_prompt(prompt_md, solution)})
        self_test_generation_attempts += 1
        try:
            tests_reply = client.chat(messages)
        except Exception as e:
            record.final_error_type = _model_error(e)
            _score_and_finalize(
                task_dir,
                sandbox,
                cfg,
                record,
                iters,
                last_public,
                last_self,
                self_tests_present,
                ran_public_tests,
                ran_self_tests,
                iteration_trace,
                last_hidden_probe,
            )
            return record
        record.tokens_in += tests_reply.tokens_in
        record.tokens_out += tests_reply.tokens_out
        messages.append({"role": "assistant", "content": tests_reply.text})

        test_files = extract_files(tests_reply.text)
        if "self_tests.py" in test_files:
            sandbox.write("self_tests.py", test_files["self_tests.py"])
            self_tests_present = True
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous response did not contain a parseable self_tests.py. "
                        "Output exactly one fenced Python block whose first line is `# self_tests.py`. "
                        "Do not include solution.py or prose."
                    ),
                }
            )
            self_test_generation_attempts += 1
            try:
                retry_reply = client.chat(messages)
            except Exception as e:
                record.final_error_type = _model_error(e)
                _score_and_finalize(
                    task_dir,
                    sandbox,
                    cfg,
                    record,
                    iters,
                    last_public,
                    last_self,
                    self_tests_present,
                    ran_public_tests,
                    ran_self_tests,
                    iteration_trace,
                    last_hidden_probe,
                )
                return record
            record.tokens_in += retry_reply.tokens_in
            record.tokens_out += retry_reply.tokens_out
            messages.append({"role": "assistant", "content": retry_reply.text})
            retry_files = extract_files(retry_reply.text)
            if "self_tests.py" in retry_files:
                sandbox.write("self_tests.py", retry_files["self_tests.py"])
                self_tests_present = True

        record.extra["self_test_generation_attempts"] = self_test_generation_attempts

        for step in range(cfg.k):
            iters = step + 1
            public_res: Optional[RunResult] = None
            self_res: Optional[RunResult] = None

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
            _save_iteration_artifacts(
                cfg,
                record,
                sandbox,
                iters,
                public_res,
                self_res,
                last_hidden_probe,
            )

            if not has_self:
                break
            if _visible_passed(cfg.mode, public_res, self_res):
                break
            if step == cfg.k - 1:
                break

            solution = sandbox.read("solution.py")
            self_tests = sandbox.read("self_tests.py") if has_self else ""
            messages.append(
                {
                    "role": "user",
                    "content": separated_solution_repair_prompt(
                        prompt_md,
                        solution,
                        self_tests,
                        self_res,
                    ),
                }
            )
            try:
                repair_reply = client.chat(messages)
            except Exception as e:
                record.final_error_type = _model_error(e)
                break
            record.tokens_in += repair_reply.tokens_in
            record.tokens_out += repair_reply.tokens_out
            messages.append({"role": "assistant", "content": repair_reply.text})

            repair_files = extract_files(repair_reply.text)
            if "solution.py" in repair_files:
                sandbox.write("solution.py", repair_files["solution.py"])

        _score_and_finalize(
            task_dir,
            sandbox,
            cfg,
            record,
            iters,
            last_public,
            last_self,
            self_tests_present,
            ran_public_tests,
            ran_self_tests,
            iteration_trace,
            last_hidden_probe,
        )
    finally:
        record.wall_time_s = round(now() - t0, 3)
        sandbox.cleanup()
    return record


def _run_task_dual(
    code_client: OllamaClient,
    test_client: OllamaClient,
    task_dir: Path,
    cfg: AgentConfig,
    record: RunRecord,
    validator_client: Optional[OllamaClient] = None,
) -> RunRecord:
    """Mode D_dual/D_val: code model repairs against frozen self-tests."""
    sandbox = Sandbox(task_dir)
    t0 = now()
    try:
        prompt_md = (task_dir / "prompt.md").read_text()
        starter = sandbox.read("solution.py")
        code_skill = _read_skill(Path(cfg.code_skill_path))
        test_skill = _read_skill(Path(cfg.test_skill_path))
        validator_skill = (
            _read_skill(Path(cfg.validator_skill_path))
            if cfg.mode == "D_val"
            else None
        )

        code_messages = [
            {"role": "system", "content": _system_with_skill(code_skill)},
            {"role": "user", "content": dual_code_initial_prompt(prompt_md, starter)},
        ]

        last_public: Optional[RunResult] = None
        last_self: Optional[RunResult] = None
        iters = 0
        self_tests_present = False
        ran_public_tests = False
        ran_self_tests = False
        iteration_trace: list[dict] = []
        last_hidden_probe: Optional[RunResult] = None
        self_test_generation_attempts = 0
        code_tokens_in = 0
        code_tokens_out = 0
        test_tokens_in = 0
        test_tokens_out = 0
        validator_tokens_in = 0
        validator_tokens_out = 0
        self_tests_validated = cfg.mode != "D_val"
        validation_attempts = 0
        validation_reason = ""
        reference_self_tests_available: Optional[bool] = None
        reference_self_tests_passed: Optional[bool] = None
        reference_self_tests_result: Optional[RunResult] = None

        record.extra["dual_model"] = True
        record.extra["code_model"] = getattr(code_client, "model", None)
        record.extra["test_model"] = getattr(test_client, "model", None)
        record.extra["code_skill_path"] = str(cfg.code_skill_path)
        record.extra["test_skill_path"] = str(cfg.test_skill_path)
        record.extra["self_tests_separate_call"] = True
        record.extra["self_tests_frozen_after_generation"] = True
        record.extra["repair_model_role"] = "code"
        if cfg.mode == "D_val":
            record.extra["validated_self_tests"] = True
            record.extra["validator_model"] = getattr(validator_client, "model", None)
            record.extra["validator_skill_path"] = str(cfg.validator_skill_path)

        try:
            solution_reply = code_client.chat(code_messages)
        except Exception as e:
            record.final_error_type = _model_error(e, "code")
            _score_and_finalize(
                task_dir,
                sandbox,
                cfg,
                record,
                iters,
                last_public,
                last_self,
                self_tests_present,
                ran_public_tests,
                ran_self_tests,
                iteration_trace,
                last_hidden_probe,
            )
            return record
        code_tokens_in += solution_reply.tokens_in
        code_tokens_out += solution_reply.tokens_out
        record.tokens_in += solution_reply.tokens_in
        record.tokens_out += solution_reply.tokens_out
        code_messages.append({"role": "assistant", "content": solution_reply.text})

        files = extract_files(solution_reply.text)
        if "solution.py" in files:
            sandbox.write("solution.py", files["solution.py"])

        solution = sandbox.read("solution.py")
        test_messages = [
            {"role": "system", "content": _system_with_skill(test_skill)},
            {"role": "user", "content": dual_test_user_prompt(prompt_md, solution)},
        ]

        self_test_generation_attempts += 1
        try:
            tests_reply = test_client.chat(test_messages)
        except Exception as e:
            record.final_error_type = _model_error(e, "test")
            _score_and_finalize(
                task_dir,
                sandbox,
                cfg,
                record,
                iters,
                last_public,
                last_self,
                self_tests_present,
                ran_public_tests,
                ran_self_tests,
                iteration_trace,
                last_hidden_probe,
            )
            return record
        test_tokens_in += tests_reply.tokens_in
        test_tokens_out += tests_reply.tokens_out
        record.tokens_in += tests_reply.tokens_in
        record.tokens_out += tests_reply.tokens_out
        test_messages.append({"role": "assistant", "content": tests_reply.text})

        test_files = extract_files(tests_reply.text)
        if "self_tests.py" in test_files:
            sandbox.write("self_tests.py", test_files["self_tests.py"])
            self_tests_present = True
        else:
            test_messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous response did not contain a parseable self_tests.py. "
                        "Output exactly one fenced Python block whose first line is `# self_tests.py`. "
                        "Do not include solution.py or prose."
                    ),
                }
            )
            self_test_generation_attempts += 1
            try:
                retry_reply = test_client.chat(test_messages)
            except Exception as e:
                record.final_error_type = _model_error(e, "test")
                _score_and_finalize(
                    task_dir,
                    sandbox,
                    cfg,
                    record,
                    iters,
                    last_public,
                    last_self,
                    self_tests_present,
                    ran_public_tests,
                    ran_self_tests,
                    iteration_trace,
                    last_hidden_probe,
                )
                return record
            test_tokens_in += retry_reply.tokens_in
            test_tokens_out += retry_reply.tokens_out
            record.tokens_in += retry_reply.tokens_in
            record.tokens_out += retry_reply.tokens_out
            test_messages.append({"role": "assistant", "content": retry_reply.text})
            retry_files = extract_files(retry_reply.text)
            if "self_tests.py" in retry_files:
                sandbox.write("self_tests.py", retry_files["self_tests.py"])
                self_tests_present = True

        if cfg.mode == "D_val":
            if validator_client is None or validator_skill is None:
                raise ValueError("mode D_val requires a validator_client and validator skill")

            for attempt in range(1, MAX_TEST_VALIDATION_ATTEMPTS + 1):
                validation_attempts = attempt
                if not self_tests_present or not (sandbox.tmp / "self_tests.py").exists():
                    validation_reason = "self_tests.py was not provided by the test model."
                    break

                solution = sandbox.read("solution.py")
                self_tests = sandbox.read("self_tests.py")
                validation_messages = [
                    {"role": "system", "content": _system_with_skill(validator_skill)},
                    {
                        "role": "user",
                        "content": test_validation_user_prompt(prompt_md, solution, self_tests),
                    },
                ]
                try:
                    validation_reply = validator_client.chat(validation_messages)
                except Exception as e:
                    record.final_error_type = _model_error(e, "validator")
                    _score_and_finalize(
                        task_dir,
                        sandbox,
                        cfg,
                        record,
                        iters,
                        last_public,
                        last_self,
                        self_tests_present,
                        ran_public_tests,
                        ran_self_tests,
                        iteration_trace,
                        last_hidden_probe,
                    )
                    return record

                validator_tokens_in += validation_reply.tokens_in
                validator_tokens_out += validation_reply.tokens_out
                record.tokens_in += validation_reply.tokens_in
                record.tokens_out += validation_reply.tokens_out
                self_tests_validated, validation_reason = _parse_test_validation(validation_reply.text)
                if self_tests_validated:
                    break
                if attempt == MAX_TEST_VALIDATION_ATTEMPTS:
                    break

                test_messages.append(
                    {
                        "role": "user",
                        "content": self_test_revision_prompt(
                            prompt_md,
                            solution,
                            self_tests,
                            validation_reason,
                        ),
                    }
                )
                self_test_generation_attempts += 1
                try:
                    revision_reply = test_client.chat(test_messages)
                except Exception as e:
                    record.final_error_type = _model_error(e, "test")
                    _score_and_finalize(
                        task_dir,
                        sandbox,
                        cfg,
                        record,
                        iters,
                        last_public,
                        last_self,
                        self_tests_present,
                        ran_public_tests,
                        ran_self_tests,
                        iteration_trace,
                        last_hidden_probe,
                    )
                    return record
                test_tokens_in += revision_reply.tokens_in
                test_tokens_out += revision_reply.tokens_out
                record.tokens_in += revision_reply.tokens_in
                record.tokens_out += revision_reply.tokens_out
                test_messages.append({"role": "assistant", "content": revision_reply.text})
                revision_files = extract_files(revision_reply.text)
                if "self_tests.py" in revision_files:
                    sandbox.write("self_tests.py", revision_files["self_tests.py"])
                    self_tests_present = True
                else:
                    self_tests_present = False
                    validation_reason = "Test writer revision did not contain a parseable self_tests.py."
                    break

            if self_tests_validated:
                reference_self_tests_result = score_self_tests_on_reference(
                    task_dir,
                    sandbox,
                    timeout=cfg.pytest_timeout,
                )
                reference_self_tests_available = reference_self_tests_result is not None
                if reference_self_tests_result is not None:
                    reference_self_tests_passed = reference_self_tests_result.returncode == 0
                    if not reference_self_tests_passed:
                        self_tests_validated = False
                        validation_reason = (
                            "self_tests.py failed against trusted reference_solution.py.\n\n"
                            + _format_pytest_result(
                                "self_tests.py on reference_solution.py",
                                reference_self_tests_result,
                            )
                        )
                else:
                    reference_self_tests_passed = None

        record.extra["self_test_generation_attempts"] = self_test_generation_attempts
        record.extra["self_tests_validated"] = self_tests_validated if cfg.mode == "D_val" else None
        record.extra["self_test_validation_attempts"] = validation_attempts if cfg.mode == "D_val" else None
        record.extra["self_test_validation_reason"] = validation_reason if cfg.mode == "D_val" else None
        if cfg.mode == "D_val":
            record.extra["reference_self_tests_available"] = reference_self_tests_available
            record.extra["reference_self_tests_passed"] = reference_self_tests_passed

        if cfg.mode == "D_val" and not self_tests_validated:
            last_self = RunResult(
                2,
                "",
                "self_tests.py was rejected by the validator: " + (validation_reason or "unknown reason"),
                False,
            )
            record.extra["code_tokens_in"] = code_tokens_in
            record.extra["code_tokens_out"] = code_tokens_out
            record.extra["test_tokens_in"] = test_tokens_in
            record.extra["test_tokens_out"] = test_tokens_out
            record.extra["validator_tokens_in"] = validator_tokens_in
            record.extra["validator_tokens_out"] = validator_tokens_out
            _save_iteration_artifacts(
                cfg,
                record,
                sandbox,
                0,
                last_public,
                last_self,
                last_hidden_probe,
            )
            _score_and_finalize(
                task_dir,
                sandbox,
                cfg,
                record,
                iters,
                last_public,
                last_self,
                self_tests_present,
                ran_public_tests,
                ran_self_tests,
                iteration_trace,
                last_hidden_probe,
            )
            if record.passed_hidden is False:
                record.final_error_type = "invalid_self_tests"
            return record

        for step in range(cfg.k):
            iters = step + 1
            public_res: Optional[RunResult] = None
            self_res: Optional[RunResult] = None

            has_self = (sandbox.tmp / "self_tests.py").exists()
            if has_self and sandbox.bash_calls < cfg.max_bash_calls:
                self_res = sandbox.run_pytest("self_tests.py", cfg.pytest_timeout)
                last_self = self_res
                self_tests_present = True
                ran_self_tests = True
            elif not has_self:
                self_res = RunResult(2, "", "self_tests.py was not provided by the test model.", False)
                last_self = self_res
            else:
                self_res = RunResult(124, "", "bash call budget exhausted before self_tests.py", False)
                last_self = self_res

            if cfg.score_hidden_each_iter:
                last_hidden_probe = score_hidden(task_dir, sandbox, timeout=cfg.hidden_timeout)
                iteration_trace.append(
                    _trace_hidden_result(iters, cfg.mode, public_res, self_res, last_hidden_probe)
                )
            _save_iteration_artifacts(
                cfg,
                record,
                sandbox,
                iters,
                public_res,
                self_res,
                last_hidden_probe,
            )

            if not has_self:
                break
            if _visible_passed(cfg.mode, public_res, self_res):
                break
            if step == cfg.k - 1:
                break

            solution = sandbox.read("solution.py")
            self_tests = sandbox.read("self_tests.py")
            repair_prompt = (
                validated_code_repair_prompt(prompt_md, solution, self_res, self_tests)
                if cfg.mode == "D_val"
                else dual_code_repair_prompt(prompt_md, solution, self_res, self_tests)
            )
            code_messages.append(
                {
                    "role": "user",
                    "content": repair_prompt,
                }
            )
            try:
                repair_reply = code_client.chat(code_messages)
            except Exception as e:
                record.final_error_type = _model_error(e, "code")
                break
            code_tokens_in += repair_reply.tokens_in
            code_tokens_out += repair_reply.tokens_out
            record.tokens_in += repair_reply.tokens_in
            record.tokens_out += repair_reply.tokens_out
            code_messages.append({"role": "assistant", "content": repair_reply.text})

            repair_files = extract_files(repair_reply.text)
            if "solution.py" in repair_files:
                sandbox.write("solution.py", repair_files["solution.py"])

        record.extra["code_tokens_in"] = code_tokens_in
        record.extra["code_tokens_out"] = code_tokens_out
        record.extra["test_tokens_in"] = test_tokens_in
        record.extra["test_tokens_out"] = test_tokens_out
        if cfg.mode == "D_val":
            record.extra["validator_tokens_in"] = validator_tokens_in
            record.extra["validator_tokens_out"] = validator_tokens_out
        _score_and_finalize(
            task_dir,
            sandbox,
            cfg,
            record,
            iters,
            last_public,
            last_self,
            self_tests_present,
            ran_public_tests,
            ran_self_tests,
            iteration_trace,
            last_hidden_probe,
        )
    finally:
        record.wall_time_s = round(now() - t0, 3)
        sandbox.cleanup()
    return record


def run_task(
    client: OllamaClient,
    task_dir: Path,
    cfg: AgentConfig,
    record: RunRecord,
    test_client: Optional[OllamaClient] = None,
    validator_client: Optional[OllamaClient] = None,
) -> RunRecord:
    if cfg.mode not in MODES:
        raise ValueError(f"unknown mode: {cfg.mode}")
    if cfg.mode in ("A", "B") and cfg.k != 1:
        raise ValueError(f"mode {cfg.mode} requires k=1")
    if cfg.mode in ("D_dual", "D_val"):
        if test_client is None:
            raise ValueError(f"mode {cfg.mode} requires a test_client")
        if cfg.mode == "D_val" and validator_client is None:
            raise ValueError("mode D_val requires a validator_client")
        return _run_task_dual(client, test_client, task_dir, cfg, record, validator_client)
    if cfg.mode == "D_sep":
        return _run_task_d_sep(client, task_dir, cfg, record)

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
                record.final_error_type = _model_error(e)
                break
            record.tokens_in += reply.tokens_in
            record.tokens_out += reply.tokens_out
            messages.append({"role": "assistant", "content": reply.text})

            files = extract_files(reply.text)
            if "solution.py" in files:
                sandbox.write("solution.py", files["solution.py"])
            if cfg.mode in SELF_TEST_MODES and "self_tests.py" in files:
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
            if cfg.mode in EXEC_SELF_TEST_MODES:
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
            _save_iteration_artifacts(
                cfg,
                record,
                sandbox,
                iters,
                public_res,
                self_res,
                last_hidden_probe,
            )

            # Non-exec modes: stop after first generation.
            if cfg.mode in ("A", "B"):
                break

            visible_pass = _visible_passed(cfg.mode, public_res, self_res)
            if visible_pass:
                break
            if step == cfg.k - 1:
                break  # no point asking again, we won't run another iter

            current_solution = sandbox.read("solution.py")
            current_self_tests = (
                sandbox.read("self_tests.py")
                if (sandbox.tmp / "self_tests.py").exists()
                else None
            )
            messages.append(
                {
                    "role": "user",
                    "content": feedback_prompt(
                        public_res,
                        self_res,
                        current_solution,
                        current_self_tests,
                    ),
                }
            )

        _score_and_finalize(
            task_dir,
            sandbox,
            cfg,
            record,
            iters,
            last_public,
            last_self,
            self_tests_present,
            ran_public_tests,
            ran_self_tests,
            iteration_trace,
            last_hidden_probe,
        )
    finally:
        record.wall_time_s = round(now() - t0, 3)
        sandbox.cleanup()
    return record
