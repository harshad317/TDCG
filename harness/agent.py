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

import ast
import copy
import re
import runpy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .log import RunRecord, now
from .models import ChatResult, OllamaClient
from .sandbox import RunResult, Sandbox, score_hidden, score_self_tests_on_reference


MODES = {"A", "B", "C", "D", "D_sep", "D_dual", "D_val", "E"}
MAX_BASH_CALLS = 20
DEFAULT_REPAIR_CANDIDATES = 3
SELF_TEST_MODES = {"B", "D", "D_sep", "D_dual", "D_val", "E"}
EXEC_SELF_TEST_MODES = {"D", "D_sep", "D_dual", "D_val", "E"}
DEFAULT_CODE_SKILL_PATH = Path("agent_skills/code_writer/skills.md")
DEFAULT_TEST_SKILL_PATH = Path("agent_skills/test_writer/skills.md")
DEFAULT_VALIDATOR_SKILL_PATH = Path("agent_skills/test_validator/skills.md")
MAX_TEST_VALIDATION_ATTEMPTS = 2
MAX_UNCHANGED_REPAIR_ESCALATIONS = 2
DEFAULT_SELF_TEST_CANDIDATES = 1
DEFAULT_CODE_CANDIDATES = 1
DEFAULT_ABORT_REPAIR_ON_MODEL_ERROR = True

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
    repair_candidates: int = DEFAULT_REPAIR_CANDIDATES
    self_test_candidates: int = DEFAULT_SELF_TEST_CANDIDATES
    code_candidates: int = DEFAULT_CODE_CANDIDATES
    abort_repair_on_model_error: bool = DEFAULT_ABORT_REPAIR_ON_MODEL_ERROR


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
            "Treat public_tests.py as a basic smoke suite only. Passing it is not proof "
            "the implementation is correct; the final answer must satisfy the full task "
            "docstring, examples, and edge cases.",
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


def dual_code_candidate_prompt(
    prompt_md: str,
    starter: str,
    candidate_index: int,
    strategy: str,
) -> str:
    return "\n".join(
        [
            f"Independent code candidate #{candidate_index}.",
            f"Strategy: {strategy}.",
            "Write the best final solution.py for the task. Do not write tests.",
            "Use a materially independent implementation, not a paraphrase of another candidate.",
            "",
            "Task description:",
            prompt_md.strip(),
            "",
            "Current solution.py starter:",
            "```python",
            starter.strip(),
            "```",
            "",
            "Output exactly one fenced Python block whose first line is `# solution.py`.",
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


TEST_STRATEGIES = [
    "examples and small edge cases",
    "boundary values, empty inputs, and single-element inputs",
    "adversarial cases that distinguish plausible wrong algorithms",
    "type-sensitive cases and duplicate-value cases",
    "larger structured cases with manually derived expected values",
]


def dual_test_candidate_user_prompt(
    prompt_md: str,
    solution: str,
    candidate_index: int,
    strategy: str,
) -> str:
    return "\n".join(
        [
            f"Independent self-test candidate #{candidate_index}.",
            f"Focus: {strategy}.",
            "Write self_tests.py to check whether solution.py satisfies the task prompt.",
            "Manually derive expected values from the prompt. Do not copy another suite.",
            "Prefer tests that catch a wrong algorithm, not only format mistakes.",
            "",
            "Task description:",
            prompt_md.strip(),
            "",
            "Current solution.py:",
            "```python",
            solution.strip(),
            "```",
            "",
            "Output exactly one fenced Python block whose first line is `# self_tests.py`.",
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
    public_res: RunResult | None = None,
) -> str:
    parts = [
        "Two-model protocol, code repair step:",
        "Task description:",
        prompt_md.strip(),
        "Current solution.py:",
        "```python\n" + solution.strip() + "\n```",
        "Frozen self_tests.py:",
        "```python\n" + self_tests.strip() + "\n```",
    ]
    if public_res is not None:
        parts.append(_format_pytest_result("public_tests.py", public_res))
    parts += [
        _format_pytest_result("self_tests.py", self_res),
        "Debug the failure above. Compare the actual value against the expected value, "
        "identify the smallest algorithmic mistake, and update solution.py only.",
        "Do not return the same implementation after a failing test run. If the actual and expected "
        "values are almost identical but one contains an extra/missing duplicated boundary item, "
        "look for an off-by-one error in a range bound, slice, or loop direction.",
        "For tasks that ask for the longest/shortest prefix, postfix, suffix, or window satisfying "
        "a condition, verify that the loop tests real candidates in the intended order and does not "
        "accidentally accept the empty candidate before a better non-empty one.",
        "For numeric transforms, preserve the prompt's exact transformation order as much as possible "
        "instead of algebraically rearranging it, because floating-point rounding can change results.",
        "The repaired solution must satisfy the task prompt, not merely hard-code these tests.",
        "Do not modify or rewrite self_tests.py.",
        "Output the full solution.py file in one fenced Python block with `# solution.py` as the first line.",
    ]
    return "\n\n".join(parts)


def validated_code_repair_prompt(
    prompt_md: str,
    solution: str,
    self_res: RunResult,
    self_tests: str,
    diagnosis: str | None = None,
    public_res: RunResult | None = None,
) -> str:
    parts = [
        "Validated self-test protocol, code repair step:",
        "Task description:",
        prompt_md.strip(),
        "Current solution.py:",
        "```python\n" + solution.strip() + "\n```",
        "Frozen validator-approved self_tests.py:",
        "```python\n" + self_tests.strip() + "\n```",
    ]
    if public_res is not None:
        parts.append(_format_pytest_result("public_tests.py", public_res))
    parts.append(_format_pytest_result("self_tests.py", self_res))
    if diagnosis:
        parts += [
            "Independent failure diagnosis:",
            diagnosis.strip(),
        ]
    parts += [
        "Debug the failure above. Compare the actual value against the expected value, "
        "identify the smallest algorithmic mistake, and update solution.py only.",
        "Do not return the same implementation after a failing test run. If the actual and expected "
        "values are almost identical but one contains an extra/missing duplicated boundary item, "
        "look for an off-by-one error in a range bound, slice, or loop direction.",
        "For tasks that ask for the longest/shortest prefix, postfix, suffix, or window satisfying "
        "a condition, verify that the loop tests real candidates in the intended order and does not "
        "accidentally accept the empty candidate before a better non-empty one.",
        "For numeric transforms, preserve the prompt's exact transformation order as much as possible "
        "instead of algebraically rearranging it, because floating-point rounding can change results.",
        "The repaired solution must satisfy the task prompt, not merely hard-code these tests.",
        "Do not modify or rewrite self_tests.py.",
        "Output the full solution.py file in one fenced Python block with `# solution.py` as the first line.",
    ]
    return "\n\n".join(parts)


def repair_diagnosis_prompt(
    prompt_md: str,
    solution: str,
    self_res: RunResult,
    self_tests: str,
    public_res: RunResult | None = None,
) -> str:
    parts = [
        "Diagnose why solution.py failed the approved pytest output.",
        "Use the task prompt as the source of truth. Do not write code.",
        "Return two concise lines:",
        "BUG: <the smallest likely bug>",
        "FIX: <the smallest required code change>",
        "Task description:",
        prompt_md.strip(),
        "Current solution.py:",
        "```python\n" + solution.strip() + "\n```",
        "Approved self_tests.py:",
        "```python\n" + self_tests.strip() + "\n```",
    ]
    if public_res is not None:
        parts.append(_format_pytest_result("public_tests.py", public_res))
    parts.append(_format_pytest_result("self_tests.py", self_res))
    return "\n\n".join(parts)


def forced_code_repair_prompt(
    prompt_md: str,
    solution: str,
    self_res: RunResult,
    self_tests: str,
    previous_reply: str,
    diagnosis: str | None = None,
    unchanged_attempt: int = 1,
    public_res: RunResult | None = None,
) -> str:
    parts = [
        f"Unchanged repair escalation #{unchanged_attempt}.",
        "The previous repair response produced no parseable code change: the parsed solution.py was identical to the failing source.",
        "The current solution is still failing tests. Output a materially different solution.py.",
        "Do not repeat the same code, control flow, loop bounds, or slice expression if they caused the observed failure.",
        "Do not explain. Do not output tests.",
        "Task description:",
        prompt_md.strip(),
        "Line-numbered current failing solution.py:",
        "```text\n" + _line_numbered(solution) + "\n```",
        "Current self_tests.py used for repair:",
        "```python\n" + self_tests.strip() + "\n```",
    ]
    if public_res is not None:
        parts.append(_format_pytest_result("public_tests.py", public_res))
    parts.append(_format_pytest_result("self_tests.py", self_res))
    if diagnosis:
        parts += [
            "Independent failure diagnosis:",
            diagnosis.strip(),
        ]
    parts += [
        "General repair checklist:",
        "- Identify the exact expression that creates the actual value shown in pytest.",
        "- If actual contains duplicated, skipped, extra, or missing boundary data, change the range bound, slice, or candidate scan order.",
        "- For longest/shortest prefix, postfix, suffix, or window tasks, test candidates in the required order and do not accept an empty candidate before a better non-empty one.",
        "- For filtering/type tasks, preserve bool/int/string distinctions.",
        "- For numeric transforms, preserve the prompt's operation order instead of relying on algebraic rewrites.",
        "Previous unusable repair response:",
        "```text\n" + previous_reply.strip()[:2000] + "\n```",
        "Output exactly one fenced Python block whose first line is `# solution.py`.",
        "The harness will ignore this response if parsed solution.py is still identical to the current failing source.",
    ]
    return "\n\n".join(parts)


def rewrite_code_repair_prompt(
    prompt_md: str,
    solution: str,
    self_res: RunResult,
    self_tests: str,
    diagnosis: str | None = None,
    public_res: RunResult | None = None,
) -> str:
    parts = [
        "Fresh rewrite repair step.",
        "A previous changed repair still failed the tests. Stop patching the current algorithm.",
        "Reimplement solution.py from the task prompt and required signatures, using the failing tests only as constraints.",
        "You may replace helper bodies and control flow. Do not preserve a loop, slice, or branch merely because it appeared in the current code.",
        "Do not explain. Do not output tests.",
        "Task description:",
        prompt_md.strip(),
        "Line-numbered current still-failing solution.py:",
        "```text\n" + _line_numbered(solution) + "\n```",
        "Current self_tests.py used for repair:",
        "```python\n" + self_tests.strip() + "\n```",
    ]
    if public_res is not None:
        parts.append(_format_pytest_result("public_tests.py", public_res))
    parts.append(_format_pytest_result("self_tests.py", self_res))
    if diagnosis:
        parts += [
            "Independent failure diagnosis:",
            diagnosis.strip(),
        ]
    parts += [
        "Rewrite checklist:",
        "- Preserve public function/class names and signatures from the starter/task.",
        "- Re-read every example in the prompt and make the implementation satisfy the general rule, not just one assertion.",
        "- For search problems, explicitly choose the candidate scan direction that matches longest/shortest/prefix/suffix wording.",
        "- For sequence construction, check whether the expected output extends the input on the left, right, or both.",
        "- For numeric transforms, preserve the prompt's operation order.",
        "Output exactly one fenced Python block whose first line is `# solution.py`.",
    ]
    return "\n\n".join(parts)


REPAIR_STRATEGIES = [
    "minimal patch of the smallest failing expression",
    "fresh direct implementation from the prompt",
    "alternate algorithm or scan order",
    "edge-case driven rewrite using the failed assertions",
    "simple brute-force or specification-first implementation when feasible",
]


def candidate_code_repair_prompt(
    prompt_md: str,
    solution: str,
    self_res: RunResult,
    self_tests: str,
    candidate_index: int,
    strategy: str,
    diagnosis: str | None = None,
    public_res: RunResult | None = None,
) -> str:
    parts = [
        f"Independent repair candidate #{candidate_index}.",
        f"Strategy: {strategy}.",
        "Generate a materially different candidate solution.py for the same task.",
        "Use the task prompt as source of truth and use the visible pytest output only as constraints.",
        "Do not hard-code the visible tests. Do not output tests. Do not explain.",
        "Task description:",
        prompt_md.strip(),
        "Line-numbered current failing solution.py:",
        "```text\n" + _line_numbered(solution) + "\n```",
        "Frozen self_tests.py:",
        "```python\n" + self_tests.strip() + "\n```",
    ]
    if public_res is not None:
        parts.append(_format_pytest_result("public_tests.py", public_res))
    parts.append(_format_pytest_result("self_tests.py", self_res))
    if diagnosis:
        parts += [
            "Independent failure diagnosis:",
            diagnosis.strip(),
        ]
    parts += [
        "Candidate requirements:",
        "- Preserve the public function/class names and signatures.",
        "- Change the algorithm, loop bounds, slice expressions, or case handling that could explain the failure.",
        "- Cover the general rule in the prompt, not only the shown assertions.",
        "- Output exactly one fenced Python block whose first line is `# solution.py`.",
    ]
    return "\n\n".join(parts)


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
            "Do not return the same implementation after a failing test run. If the actual and expected "
            "values are almost identical but one contains an extra/missing duplicated boundary item, "
            "look for an off-by-one error in a range bound, slice, or loop direction.",
            "For tasks that ask for the longest/shortest prefix, postfix, suffix, or window satisfying "
            "a condition, verify that the loop tests real candidates in the intended order and does not "
            "accidentally accept the empty candidate before a better non-empty one.",
            "For numeric transforms, preserve the prompt's exact transformation order as much as possible "
            "instead of algebraically rearranging it, because floating-point rounding can change results.",
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


def _compact_repair_diagnosis(text: str, max_chars: int = 800) -> str:
    selected = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"(?i)^\s*(BUG|FIX)\s*:", line)
    ]
    compact = "\n".join(selected).strip() or text.strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "\n... [truncated]"


def _line_numbered(text: str, max_chars: int = 5000) -> str:
    numbered = "\n".join(
        f"{i:04d}: {line}" for i, line in enumerate(text.splitlines(), start=1)
    )
    if len(numbered) <= max_chars:
        return numbered
    return numbered[:max_chars] + "\n... [truncated]"


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
    if mode in ("C", "D_val", "E"):
        required.append(public_res)
    if mode in EXEC_SELF_TEST_MODES:
        required.append(self_res)
    return bool(required) and all(res is not None and res.returncode == 0 for res in required)


def _visible_test_files(sandbox: Sandbox, include_public: bool) -> list[str]:
    files: list[str] = []
    if include_public and (sandbox.tmp / "public_tests.py").exists():
        files.append("public_tests.py")
    if (sandbox.tmp / "self_tests.py").exists():
        files.append("self_tests.py")
    return files


def _run_candidate_visible_tests(
    sandbox: Sandbox,
    cfg: AgentConfig,
    *,
    include_public: bool,
) -> RunResult:
    files = _visible_test_files(sandbox, include_public)
    if not files:
        return RunResult(2, "", "no visible test files available for candidate scoring", False)
    if sandbox.bash_calls >= cfg.max_bash_calls:
        return RunResult(124, "", "bash call budget exhausted before candidate visible tests", False)
    return sandbox.run_pytest_files(files, cfg.pytest_timeout)


def _visible_failure_count(res: RunResult) -> int:
    if res.returncode == 0:
        return 0
    if res.timed_out or res.returncode == 124:
        return 1_000_000
    output = res.stdout + "\n" + res.stderr
    failed_counts = [int(n) for n in re.findall(r"(\d+)\s+failed\b", output)]
    error_counts = [int(n) for n in re.findall(r"(\d+)\s+errors?\b", output)]
    if failed_counts or error_counts:
        return sum(failed_counts) + sum(error_counts)
    return 999_999


def _result_passed(res: Optional[RunResult]) -> Optional[bool]:
    return (res.returncode == 0) if res is not None else None


def _model_error(e: Exception, role: str = "model") -> str:
    message = str(e).replace("\n", " ").strip()
    if len(message) > 160:
        message = message[:157] + "..."
    return f"model_error:{role}:{type(e).__name__}" + (f":{message}" if message else "")


def _is_timeout_error(e: Exception) -> bool:
    if isinstance(e, TimeoutError):
        return True
    text = f"{type(e).__name__}: {e}".lower()
    return "timeout" in text or "timed out" in text


def _parse_test_validation(text: str) -> tuple[bool, str]:
    match = re.search(r"(?im)^\s*TESTS_VALID\s*:\s*(yes|no|true|false)\s*$", text)
    valid = bool(match and match.group(1).lower() in {"yes", "true"})
    reason_match = re.search(r"(?ims)^\s*REASON\s*:\s*(.*)$", text)
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    if len(reason) > 1200:
        reason = reason[:1197] + "..."
    return valid, reason


def _expand_self_test_assertions(source: str) -> str:
    """Split simple multi-assert pytest functions so repair sees more failures."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    changed = False
    new_body: list[ast.stmt] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            new_body.append(node)
            continue

        prefix: list[ast.stmt] = []
        asserts: list[ast.Assert] = []
        seen_assert = False
        splittable = True
        for stmt in node.body:
            if isinstance(stmt, ast.Assert):
                seen_assert = True
                asserts.append(stmt)
            elif seen_assert:
                splittable = False
                break
            else:
                prefix.append(stmt)

        if not splittable or len(asserts) <= 1:
            new_body.append(node)
            continue

        changed = True
        for idx, assert_stmt in enumerate(asserts, start=1):
            split_node = copy.deepcopy(node)
            split_node.name = f"{node.name}_{idx}"
            split_node.body = [copy.deepcopy(stmt) for stmt in prefix] + [copy.deepcopy(assert_stmt)]
            new_body.append(split_node)

    if not changed:
        return source
    tree.body = new_body
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree).rstrip() + "\n"
    except Exception:
        return source


SAFE_TEST_IMPORTS = {
    "collections",
    "dataclasses",
    "functools",
    "itertools",
    "math",
    "operator",
    "pytest",
    "re",
    "solution",
    "statistics",
    "string",
    "sys",
    "typing",
}
UNSAFE_TEST_IMPORTS = {
    "asyncio",
    "builtins",
    "ctypes",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "tempfile",
    "threading",
    "time",
    "urllib",
}
UNSAFE_TEST_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "input",
    "locals",
    "open",
    "quit",
}


def _target_symbols_from_public_tests(task_dir: Path) -> set[str]:
    public_tests = task_dir / "public_tests.py"
    if not public_tests.exists():
        return set()
    try:
        tree = ast.parse(public_tests.read_text())
    except SyntaxError:
        return set()

    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "solution":
            for alias in node.names:
                if alias.name != "*":
                    symbols.add(alias.asname or alias.name)
    return symbols


def _static_self_test_rejection_reason(task_dir: Path, self_tests: str) -> str | None:
    try:
        tree = ast.parse(self_tests)
    except SyntaxError as e:
        return f"self_tests.py is invalid Python: {e.msg} at line {e.lineno}"

    has_test_function = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in tree.body
    )
    has_assert = any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    if not has_test_function and not has_assert:
        return "self_tests.py must contain pytest tests or assertions."

    target_call_names: set[str] = set()
    solution_module_aliases = {"solution"}
    imports_solution_module = False
    imported_modules: set[str] = set()
    called_names: set[str] = set()
    attribute_calls: set[tuple[str, str]] = set()
    target_symbols = _target_symbols_from_public_tests(task_dir)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".", 1)[0]
                imported_modules.add(module)
                if module == "solution":
                    imports_solution_module = True
                    solution_module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            imported_modules.add(module)
            if node.module == "solution":
                for alias in node.names:
                    if alias.name == "*":
                        imports_solution_module = True
                    else:
                        if not target_symbols or alias.name in target_symbols:
                            target_call_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called_names.add(func.id)
            elif isinstance(func, ast.Attribute):
                base = func.value
                if isinstance(base, ast.Name):
                    attribute_calls.add((base.id, func.attr))

    unsafe_imports = sorted(module for module in imported_modules if module in UNSAFE_TEST_IMPORTS)
    if unsafe_imports:
        return "self_tests.py imports unsafe modules: " + ", ".join(unsafe_imports)

    unknown_imports = sorted(
        module for module in imported_modules
        if module not in SAFE_TEST_IMPORTS and module not in UNSAFE_TEST_IMPORTS
    )
    if unknown_imports:
        return "self_tests.py imports unsupported modules: " + ", ".join(unknown_imports)

    unsafe_calls = sorted(name for name in called_names if name in UNSAFE_TEST_CALLS)
    if unsafe_calls:
        return "self_tests.py calls unsafe builtins: " + ", ".join(unsafe_calls)

    if target_symbols:
        direct_hits = target_call_names & called_names
        module_hits = {
            attr
            for base, attr in attribute_calls
            if base in solution_module_aliases and attr in target_symbols
        }
        if not direct_hits and not module_hits:
            return (
                "self_tests.py does not import or call the benchmark entry point(s): "
                + ", ".join(sorted(target_symbols))
            )
    elif not target_call_names and not imports_solution_module:
        return "self_tests.py must import the public function/class from solution.py."

    return None


def _reference_self_test_rejection_reason(
    task_dir: Path,
    sandbox: Sandbox,
    cfg: AgentConfig,
) -> tuple[str | None, RunResult | None, bool | None, bool]:
    reference_result = score_self_tests_on_reference(
        task_dir,
        sandbox,
        timeout=cfg.pytest_timeout,
    )
    if reference_result is None:
        return None, None, None, False
    reference_passed = reference_result.returncode == 0
    if reference_passed:
        strict_reason, strict_result = _strict_reference_assertion_rejection_reason(
            task_dir,
            sandbox.read("self_tests.py"),
        )
        if strict_reason is not None:
            return strict_reason, strict_result, False, True
        return None, reference_result, True, True
    return (
        "self_tests.py failed against trusted reference_solution.py.\n\n"
        + _format_pytest_result(
            "self_tests.py on reference_solution.py",
            reference_result,
        ),
        reference_result,
        False,
        True,
    )


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "solution"
    ):
        return node.attr
    return None


def _literal_call_assertion(
    node: ast.Assert,
) -> tuple[str, list[object], dict[str, object], object] | None:
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and len(test.comparators) == 1
        and isinstance(test.ops[0], (ast.Eq, ast.Is))
    ):
        return None

    pairs = ((test.left, test.comparators[0]), (test.comparators[0], test.left))
    for call_node, expected_node in pairs:
        if not isinstance(call_node, ast.Call):
            continue
        name = _call_name(call_node.func)
        if name is None:
            continue
        try:
            args = [ast.literal_eval(arg) for arg in call_node.args]
            kwargs = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in call_node.keywords
                if kw.arg is not None
            }
            expected = ast.literal_eval(expected_node)
        except (SyntaxError, TypeError, ValueError):
            continue
        return name, args, kwargs, expected
    return None


def _strict_reference_assertion_rejection_reason(
    task_dir: Path,
    self_tests: str,
) -> tuple[str | None, RunResult | None]:
    """Catch pytest equality cases where reference passes but types are wrong.

    Pytest treats `False == 0` as true, while the EvalPlus hidden comparator
    distinguishes booleans from numbers. This extra reference check is narrow:
    it only inspects simple literal assertions and rejects bool/int-equivalent
    expected values that would otherwise pass against reference_solution.py.
    """
    reference = task_dir / "reference_solution.py"
    if not reference.exists():
        return None, None

    try:
        tree = ast.parse(self_tests)
    except SyntaxError:
        return None, None

    try:
        reference_globals = runpy.run_path(str(reference))
    except Exception as e:
        return (
            f"trusted reference_solution.py could not be loaded for strict checks: {e}",
            RunResult(1, "", str(e), False),
        )

    target_symbols = _target_symbols_from_public_tests(task_dir)
    failures: list[str] = []
    for top_level in tree.body:
        if not isinstance(top_level, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not top_level.name.startswith("test_"):
            continue
        for node in ast.walk(top_level):
            if not isinstance(node, ast.Assert):
                continue
            assertion = _literal_call_assertion(node)
            if assertion is None:
                continue
            name, args, kwargs, expected = assertion
            if target_symbols and name not in target_symbols:
                continue
            func = reference_globals.get(name)
            if not callable(func):
                continue
            try:
                actual = func(*args, **kwargs)
            except Exception:
                continue
            if not (isinstance(actual, bool) or isinstance(expected, bool)):
                continue
            if actual is expected:
                continue
            failures.append(
                "FAILED self_tests.py::"
                f"{top_level.name}\n"
                f"strict reference mismatch: {name} returned "
                f"{actual!r} ({type(actual).__name__}) for args={args!r}, "
                f"but the test expected {expected!r} ({type(expected).__name__})."
            )

    if not failures:
        return None, None

    output = "\n".join(failures)
    reason = (
        "self_tests.py passed normal pytest against reference_solution.py, but "
        "strict reference checking found bool/int-equivalent expected values.\n\n"
        + output
    )
    return reason, RunResult(1, output, "", False)


FAILED_TEST_RE = re.compile(r"FAILED\s+self_tests\.py::([A-Za-z_]\w*)\b")


def _test_function_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _drop_test_functions(source: str, names: set[str]) -> str | None:
    if not names:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines()
    drop_ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in names or not node.name.startswith("test_"):
            continue
        end_lineno = getattr(node, "end_lineno", node.lineno)
        drop_ranges.append((node.lineno, end_lineno))

    if not drop_ranges:
        return source

    keep: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        if any(start <= lineno <= end for start, end in drop_ranges):
            continue
        keep.append(line)
    return "\n".join(keep).rstrip() + "\n"


def _namespace_test_functions(source: str, prefix: str) -> str:
    """Rename pytest functions so independently generated suites can be merged."""
    try:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                node.name = f"test_{prefix}_{node.name[5:]}"
        ast.fix_missing_locations(tree)
        return ast.unparse(tree).rstrip() + "\n"
    except Exception:
        safe_prefix = re.sub(r"\W+", "_", prefix).strip("_") or "suite"
        return re.sub(
            r"(?m)^(\s*def\s+)test_([A-Za-z_]\w*)\s*\(",
            rf"\1test_{safe_prefix}_\2(",
            source,
        )


def _merge_self_test_sources(sources: list[str]) -> str:
    chunks = ["# self_tests.py", "# Merged independent validated self-test suites."]
    for idx, source in enumerate(sources, start=1):
        chunks.append("")
        chunks.append(f"# ---- validated suite {idx} ----")
        chunks.append(_namespace_test_functions(source, f"suite{idx}").strip())
    return "\n".join(chunks).rstrip() + "\n"


def _prune_self_tests_to_reference_passing(
    task_dir: Path,
    sandbox: Sandbox,
    cfg: AgentConfig,
    source: str,
) -> tuple[str | None, RunResult | None, list[str], bool]:
    """Drop only generated test functions that fail against the trusted reference."""
    working = _expand_self_test_assertions(source)
    dropped: list[str] = []
    available = False

    for _ in range(3):
        sandbox.write("self_tests.py", working)
        (
            reference_reason,
            reference_result,
            reference_passed,
            reference_available,
        ) = _reference_self_test_rejection_reason(task_dir, sandbox, cfg)
        available = reference_available
        if not reference_available:
            return None, reference_result, dropped, False
        if reference_reason is None and reference_passed:
            if _test_function_names(working):
                return working, reference_result, dropped, True
            return None, reference_result, dropped, True

        output = ""
        if reference_result is not None:
            output = reference_result.stdout + "\n" + reference_result.stderr
        failed = set(FAILED_TEST_RE.findall(output))
        tests = _test_function_names(working)
        to_drop = failed & tests
        if not to_drop or to_drop == tests:
            return None, reference_result, dropped, available

        pruned = _drop_test_functions(working, to_drop)
        if pruned is None or pruned == working:
            return None, reference_result, dropped, available
        dropped.extend(sorted(to_drop))
        working = pruned

    return None, None, dropped, available


def _validate_or_prune_self_test_source(
    task_dir: Path,
    sandbox: Sandbox,
    cfg: AgentConfig,
    prompt_md: str,
    solution: str,
    source: str,
    validator_client: Optional[OllamaClient],
    validator_skill: Optional[str],
) -> tuple[str | None, dict, int, int]:
    """Validate one generated self-test source and keep only trusted tests."""
    expanded = _expand_self_test_assertions(source)
    info: dict = {
        "accepted": False,
        "reason": "",
        "reference_available": None,
        "reference_passed": None,
        "pruned_used": False,
        "pruned_dropped": [],
        "validator_used": False,
    }

    static_reason = _static_self_test_rejection_reason(task_dir, expanded)
    if static_reason is not None:
        info["reason"] = static_reason
        return None, info, 0, 0

    sandbox.write("self_tests.py", expanded)
    (
        reference_reason,
        reference_result,
        reference_passed,
        reference_available,
    ) = _reference_self_test_rejection_reason(task_dir, sandbox, cfg)
    info["reference_available"] = reference_available
    info["reference_passed"] = reference_passed
    if reference_available and reference_reason is None and reference_passed:
        info["accepted"] = True
        info["reason"] = "self_tests.py passed trusted reference_solution.py."
        return expanded, info, 0, 0

    if reference_available and reference_reason is not None:
        pruned, _, dropped, _ = _prune_self_tests_to_reference_passing(
            task_dir,
            sandbox,
            cfg,
            expanded,
        )
        if pruned is not None:
            info["accepted"] = True
            info["reference_passed"] = True
            info["pruned_used"] = True
            info["pruned_dropped"] = dropped
            info["reason"] = "kept only generated test functions that passed trusted reference_solution.py."
            return pruned, info, 0, 0
        info["reason"] = reference_reason
        return None, info, 0, 0

    if validator_client is None or validator_skill is None:
        info["reason"] = "no reference_solution.py or validator model was available."
        return None, info, 0, 0

    validation_messages = [
        {"role": "system", "content": _system_with_skill(validator_skill)},
        {
            "role": "user",
            "content": test_validation_user_prompt(prompt_md, solution, expanded),
        },
    ]
    validation_reply = validator_client.chat(validation_messages)
    valid, reason = _parse_test_validation(validation_reply.text)
    info["validator_used"] = True
    info["accepted"] = valid
    info["reason"] = reason
    if valid:
        return expanded, info, validation_reply.tokens_in, validation_reply.tokens_out
    return None, info, validation_reply.tokens_in, validation_reply.tokens_out


def _fallback_self_tests_from_public(task_dir: Path) -> str | None:
    public_tests = task_dir / "public_tests.py"
    if not public_tests.exists():
        return None
    try:
        source = public_tests.read_text()
    except OSError:
        return None
    if not source.strip():
        return None
    return (
        "# self_tests.py\n"
        "# Fallback: use benchmark public tests after generated self-tests failed reference validation.\n"
        + source
    )


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
        record.extra["ran_public_tests"] = ran_public_tests if cfg.mode in ("C", "D_val", "E") else None
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
    record.extra["ran_public_tests"] = ran_public_tests if cfg.mode in ("C", "D_val", "E") else None
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
            elif cfg.mode == "D_val":
                record.final_error_type = "overfit_public_self"
            elif cfg.mode in ("D", "D_sep", "D_dual"):
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
        code_repair_retry_count = 0
        code_repair_unchanged_count = 0
        code_repair_stagnation_exhausted_count = 0
        code_repair_rewrite_count = 0
        previous_repair_applied = False
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
            sandbox.write("self_tests.py", _expand_self_test_assertions(test_files["self_tests.py"]))
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
                sandbox.write("self_tests.py", _expand_self_test_assertions(retry_files["self_tests.py"]))
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
            use_rewrite_repair = previous_repair_applied
            previous_repair_applied = False
            messages.append(
                {
                    "role": "user",
                    "content": (
                        rewrite_code_repair_prompt(
                            prompt_md,
                            solution,
                            self_res,
                            self_tests,
                        )
                        if use_rewrite_repair
                        else separated_solution_repair_prompt(
                            prompt_md,
                            solution,
                            self_tests,
                            self_res,
                        )
                    ),
                }
            )
            if use_rewrite_repair:
                code_repair_rewrite_count += 1
            try:
                repair_reply = client.chat(messages)
            except Exception as e:
                record.final_error_type = _model_error(e)
                break
            record.tokens_in += repair_reply.tokens_in
            record.tokens_out += repair_reply.tokens_out
            messages.append({"role": "assistant", "content": repair_reply.text})

            repair_files = extract_files(repair_reply.text)
            candidate_solution = repair_files.get("solution.py")
            if candidate_solution and candidate_solution.strip() != solution.strip():
                sandbox.write("solution.py", candidate_solution)
                previous_repair_applied = True
            else:
                code_repair_unchanged_count += 1
                previous_reply = repair_reply.text
                applied_repair = False
                repair_retry_failed = False
                for unchanged_attempt in range(1, MAX_UNCHANGED_REPAIR_ESCALATIONS + 1):
                    retry_messages = [
                        {"role": "system", "content": SYSTEM_BASE},
                        {
                            "role": "user",
                            "content": forced_code_repair_prompt(
                                prompt_md,
                                solution,
                                self_res,
                                self_tests,
                                previous_reply,
                                unchanged_attempt=unchanged_attempt,
                            ),
                        },
                    ]
                    try:
                        repair_retry_reply = client.chat(retry_messages)
                    except Exception as e:
                        record.final_error_type = _model_error(e)
                        repair_retry_failed = True
                        break
                    code_repair_retry_count += 1
                    record.tokens_in += repair_retry_reply.tokens_in
                    record.tokens_out += repair_retry_reply.tokens_out
                    messages.append(
                        {
                            "role": "user",
                            "content": "The previous repair was unchanged; a fresh repair attempt was requested.",
                        }
                    )
                    messages.append({"role": "assistant", "content": repair_retry_reply.text})
                    retry_files = extract_files(repair_retry_reply.text)
                    retry_solution = retry_files.get("solution.py")
                    if retry_solution and retry_solution.strip() != solution.strip():
                        sandbox.write("solution.py", retry_solution)
                        previous_repair_applied = True
                        applied_repair = True
                        break
                    code_repair_unchanged_count += 1
                    previous_reply = repair_retry_reply.text
                if repair_retry_failed:
                    break
                if not applied_repair:
                    code_repair_stagnation_exhausted_count += 1

        record.extra["code_repair_retry_count"] = code_repair_retry_count
        record.extra["code_repair_unchanged_count"] = code_repair_unchanged_count
        record.extra["code_repair_stagnation_exhausted_count"] = code_repair_stagnation_exhausted_count
        record.extra["code_repair_rewrite_count"] = code_repair_rewrite_count
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
    repair_client: Optional[OllamaClient] = None,
) -> RunRecord:
    """Mode D_dual/D_val: code model repairs against frozen self-tests."""
    repair_client = repair_client or code_client
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
        code_repair_retry_count = 0
        code_repair_unchanged_count = 0
        code_repair_stagnation_exhausted_count = 0
        code_repair_rewrite_count = 0
        repair_tokens_in = 0
        repair_tokens_out = 0
        repair_candidate_search_count = 0
        repair_candidates_requested_total = 0
        repair_candidates_generated_count = 0
        repair_candidates_tested_count = 0
        repair_candidates_visible_pass_count = 0
        repair_candidates_selected_visible_pass_count = 0
        repair_candidate_best_failures: list[int] = []
        repair_candidate_model_error_count = 0
        repair_model_error_count = 0
        repair_model_timeout_count = 0
        repair_fallback_to_code_count = 0
        repair_fallback_to_code_error_count = 0
        repair_fallback_to_code_timeout_count = 0
        repair_aborted_after_model_error = False
        repair_preserved_solution_count = 0
        repair_model_errors: list[str] = []
        stop_future_repairs_after_model_error = False
        previous_repair_applied = False
        repair_diagnosis_count = 0
        repair_diagnosis_tokens_in = 0
        repair_diagnosis_tokens_out = 0
        self_tests_validated = cfg.mode != "D_val"
        validation_attempts = 0
        validation_reason = ""
        reference_self_tests_available: Optional[bool] = None
        reference_self_tests_passed: Optional[bool] = None
        reference_self_tests_result: Optional[RunResult] = None
        self_tests_fallback_used = False
        self_tests_pruned_used = False
        self_tests_pruned_dropped: list[str] = []
        self_test_candidates_requested_total = max(1, cfg.self_test_candidates)
        self_test_candidates_generated_count = 0
        self_test_candidates_accepted_count = 0
        self_test_candidates_pruned_count = 0
        self_test_candidates_invalid_count = 0
        self_test_candidate_reasons: list[str] = []
        code_candidates_requested_total = max(1, cfg.code_candidates)
        code_candidates_generated_count = 0
        code_candidates_tested_count = 0
        code_candidates_visible_pass_count = 0
        code_candidates_selected_visible_pass = False
        code_candidate_best_failures: Optional[int] = None
        code_candidate_model_error_count = 0

        record.extra["dual_model"] = True
        record.extra["code_model"] = getattr(code_client, "model", None)
        record.extra["test_model"] = getattr(test_client, "model", None)
        record.extra["code_skill_path"] = str(cfg.code_skill_path)
        record.extra["test_skill_path"] = str(cfg.test_skill_path)
        record.extra["self_tests_separate_call"] = True
        record.extra["self_tests_frozen_after_generation"] = True
        record.extra["repair_model_role"] = "repair" if repair_client is not code_client else "code"
        record.extra["repair_model"] = getattr(repair_client, "model", None)
        record.extra["repair_model_separate"] = repair_client is not code_client
        record.extra["repair_candidates_config"] = cfg.repair_candidates
        record.extra["abort_repair_on_model_error"] = cfg.abort_repair_on_model_error
        record.extra["self_test_candidates_config"] = cfg.self_test_candidates
        record.extra["code_candidates_config"] = cfg.code_candidates
        if cfg.mode == "D_val":
            record.extra["validated_self_tests"] = True
            record.extra["d_val_requires_public_and_self"] = True
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
            self_test_candidates_generated_count += 1
            sandbox.write("self_tests.py", _expand_self_test_assertions(test_files["self_tests.py"]))
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
                self_test_candidates_generated_count += 1
                sandbox.write("self_tests.py", _expand_self_test_assertions(retry_files["self_tests.py"]))
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
                static_reason = _static_self_test_rejection_reason(task_dir, self_tests)
                if static_reason is not None:
                    self_tests_validated = False
                    validation_reason = static_reason
                else:
                    (
                        reference_reason,
                        reference_self_tests_result,
                        reference_self_tests_passed,
                        reference_self_tests_available,
                    ) = _reference_self_test_rejection_reason(task_dir, sandbox, cfg)
                    if reference_reason is not None:
                        self_tests_validated = False
                        validation_reason = reference_reason
                    elif reference_self_tests_available:
                        self_tests_validated = True
                        validation_reason = "self_tests.py passed trusted reference_solution.py."
                    else:
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
                    self_test_candidates_generated_count += 1
                    sandbox.write("self_tests.py", _expand_self_test_assertions(revision_files["self_tests.py"]))
                    self_tests_present = True
                else:
                    self_tests_present = False
                    validation_reason = "Test writer revision did not contain a parseable self_tests.py."
                    break

            if not self_tests_validated:
                last_validation_reason = validation_reason
                pruned_self_tests = None
                if self_tests_present and (sandbox.tmp / "self_tests.py").exists():
                    candidate_self_tests = sandbox.read("self_tests.py")
                    (
                        pruned_self_tests,
                        reference_self_tests_result,
                        self_tests_pruned_dropped,
                        reference_self_tests_available,
                    ) = _prune_self_tests_to_reference_passing(
                        task_dir,
                        sandbox,
                        cfg,
                        candidate_self_tests,
                    )
                    if pruned_self_tests is not None:
                        sandbox.write("self_tests.py", _expand_self_test_assertions(pruned_self_tests))
                        self_tests_present = True
                        self_tests_validated = True
                        self_tests_pruned_used = True
                        reference_self_tests_passed = True
                        validation_reason = (
                            "Generated self-tests failed validation; kept only generated "
                            "test functions that passed trusted reference_solution.py."
                        )
                        if self_tests_pruned_dropped:
                            validation_reason += (
                                "\nDropped invalid generated tests: "
                                + ", ".join(self_tests_pruned_dropped)
                            )
                        if last_validation_reason:
                            validation_reason += "\n\nLast generated-test rejection:\n" + last_validation_reason

                fallback_self_tests = (
                    None if self_tests_validated else _fallback_self_tests_from_public(task_dir)
                )
                if fallback_self_tests is not None:
                    sandbox.write("self_tests.py", _expand_self_test_assertions(fallback_self_tests))
                    self_tests_present = True
                    self_tests_fallback_used = True
                    fallback_static_reason = _static_self_test_rejection_reason(
                        task_dir,
                        fallback_self_tests,
                    )
                    if fallback_static_reason is not None:
                        validation_reason = (
                            (last_validation_reason + "\n\n") if last_validation_reason else ""
                        ) + "Fallback public_tests.py was rejected: " + fallback_static_reason
                    else:
                        (
                            fallback_reference_reason,
                            reference_self_tests_result,
                            reference_self_tests_passed,
                            reference_self_tests_available,
                        ) = _reference_self_test_rejection_reason(task_dir, sandbox, cfg)
                        if fallback_reference_reason is not None:
                            validation_reason = (
                                (last_validation_reason + "\n\n") if last_validation_reason else ""
                            ) + "Fallback public_tests.py failed reference validation:\n" + fallback_reference_reason
                        else:
                            self_tests_validated = True
                            if reference_self_tests_available:
                                validation_reason = (
                                    "Generated self-tests failed validation; fell back to "
                                    "public_tests.py, which passed trusted reference_solution.py."
                                )
                            else:
                                validation_reason = (
                                    "Generated self-tests failed validation; fell back to "
                                    "public_tests.py because no reference_solution.py was available."
                                )
                            if last_validation_reason:
                                validation_reason += "\n\nLast generated-test rejection:\n" + last_validation_reason

            validated_self_test_sources: list[str] = []
            if self_tests_validated and self_tests_present and (sandbox.tmp / "self_tests.py").exists():
                validated_self_test_sources.append(sandbox.read("self_tests.py"))
                self_test_candidates_accepted_count += 1
                if self_tests_pruned_used:
                    self_test_candidates_pruned_count += 1

            if self_tests_validated and cfg.self_test_candidates > 1:
                solution = sandbox.read("solution.py")
                for candidate_index in range(2, cfg.self_test_candidates + 1):
                    strategy = TEST_STRATEGIES[
                        (candidate_index - 2) % len(TEST_STRATEGIES)
                    ]
                    candidate_messages = [
                        {"role": "system", "content": _system_with_skill(test_skill)},
                        {
                            "role": "user",
                            "content": dual_test_candidate_user_prompt(
                                prompt_md,
                                solution,
                                candidate_index,
                                strategy,
                            ),
                        },
                    ]
                    self_test_generation_attempts += 1
                    try:
                        candidate_reply = test_client.chat(candidate_messages)
                    except Exception as e:
                        record.final_error_type = _model_error(e, "test")
                        break
                    test_tokens_in += candidate_reply.tokens_in
                    test_tokens_out += candidate_reply.tokens_out
                    record.tokens_in += candidate_reply.tokens_in
                    record.tokens_out += candidate_reply.tokens_out

                    candidate_files = extract_files(candidate_reply.text)
                    candidate_source = candidate_files.get("self_tests.py")
                    if candidate_source is None:
                        self_test_candidates_invalid_count += 1
                        self_test_candidate_reasons.append(
                            f"candidate_{candidate_index}: missing self_tests.py"
                        )
                        continue

                    self_test_candidates_generated_count += 1
                    try:
                        (
                            accepted_source,
                            candidate_info,
                            val_tokens_in,
                            val_tokens_out,
                        ) = _validate_or_prune_self_test_source(
                            task_dir,
                            sandbox,
                            cfg,
                            prompt_md,
                            solution,
                            candidate_source,
                            validator_client,
                            validator_skill,
                        )
                    except Exception as e:
                        record.final_error_type = _model_error(e, "validator")
                        break
                    validator_tokens_in += val_tokens_in
                    validator_tokens_out += val_tokens_out
                    record.tokens_in += val_tokens_in
                    record.tokens_out += val_tokens_out
                    if accepted_source is None:
                        self_test_candidates_invalid_count += 1
                        reason = str(candidate_info.get("reason") or "rejected")
                        self_test_candidate_reasons.append(
                            f"candidate_{candidate_index}: {reason[:240]}"
                        )
                        continue

                    validated_self_test_sources.append(accepted_source)
                    self_test_candidates_accepted_count += 1
                    if candidate_info.get("pruned_used"):
                        self_test_candidates_pruned_count += 1

                if len(validated_self_test_sources) > 1:
                    original_self_tests = validated_self_test_sources[0]
                    merged_self_tests = _merge_self_test_sources(validated_self_test_sources)
                    sandbox.write("self_tests.py", merged_self_tests)
                    merge_static_reason = _static_self_test_rejection_reason(
                        task_dir,
                        merged_self_tests,
                    )
                    merge_reference_reason = None
                    merge_reference_available = False
                    if merge_static_reason is None:
                        (
                            merge_reference_reason,
                            reference_self_tests_result,
                            reference_self_tests_passed,
                            merge_reference_available,
                        ) = _reference_self_test_rejection_reason(task_dir, sandbox, cfg)
                    if merge_static_reason is not None or merge_reference_reason is not None:
                        sandbox.write("self_tests.py", original_self_tests)
                        validation_reason += (
                            "\n\nMerged extra self-test suites were rejected; kept original suite."
                        )
                        if merge_static_reason is not None:
                            validation_reason += "\nMerge rejection: " + merge_static_reason
                        elif merge_reference_reason is not None:
                            validation_reason += "\nMerge rejection: " + merge_reference_reason[:500]
                    else:
                        self_tests_present = True
                        self_tests_validated = True
                        validation_reason += (
                            f"\n\nMerged {len(validated_self_test_sources)} validated self-test suites."
                        )
                        if merge_reference_available:
                            reference_self_tests_available = True
                            reference_self_tests_passed = True
                elif validated_self_test_sources:
                    sandbox.write("self_tests.py", validated_self_test_sources[0])

        record.extra["self_test_generation_attempts"] = self_test_generation_attempts
        record.extra["self_tests_validated"] = self_tests_validated if cfg.mode == "D_val" else None
        record.extra["self_test_validation_attempts"] = validation_attempts if cfg.mode == "D_val" else None
        record.extra["self_test_validation_reason"] = validation_reason if cfg.mode == "D_val" else None
        if cfg.mode == "D_val":
            record.extra["reference_self_tests_available"] = reference_self_tests_available
            record.extra["reference_self_tests_passed"] = reference_self_tests_passed
            record.extra["self_tests_fallback_used"] = self_tests_fallback_used
            record.extra["self_tests_pruned_used"] = self_tests_pruned_used
            record.extra["self_tests_pruned_dropped"] = self_tests_pruned_dropped
            record.extra["self_test_candidates_requested_total"] = self_test_candidates_requested_total
            record.extra["self_test_candidates_generated_count"] = self_test_candidates_generated_count
            record.extra["self_test_candidates_accepted_count"] = self_test_candidates_accepted_count
            record.extra["self_test_candidates_pruned_count"] = self_test_candidates_pruned_count
            record.extra["self_test_candidates_invalid_count"] = self_test_candidates_invalid_count
            record.extra["self_test_candidate_reasons"] = self_test_candidate_reasons[:10]

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
            record.iterations_used = iters
            record.bash_calls_used = sandbox.bash_calls
            record.passed_public = None
            record.passed_self = False
            record.passed_hidden = None
            record.final_error_type = "validator_invalid_self_tests"
            record.extra["pytest_timeout_s"] = cfg.pytest_timeout
            record.extra["hidden_timeout_s"] = cfg.hidden_timeout
            record.extra["hidden_timed_out"] = None
            record.extra["score_hidden_each_iter"] = cfg.score_hidden_each_iter
            if cfg.score_hidden_each_iter:
                record.extra["iteration_trace"] = iteration_trace
            record.extra["self_tests_present"] = self_tests_present
            record.extra["ran_public_tests"] = None
            record.extra["ran_self_tests"] = False
            record.extra["initial_hidden_pass"] = None
            record.extra["hidden_improved"] = None
            record.extra["fixed_by_self_tests"] = None
            return record

        if cfg.code_candidates > 1 and self_tests_present and (sandbox.tmp / "self_tests.py").exists():
            initial_solution = sandbox.read("solution.py")
            code_candidate_records: list[dict] = []

            def consider_code_candidate(candidate_solution: str, source: str) -> None:
                nonlocal code_candidates_generated_count
                nonlocal code_candidates_tested_count
                nonlocal code_candidates_visible_pass_count

                if not candidate_solution.strip():
                    return
                code_candidates_generated_count += 1
                sandbox.write("solution.py", candidate_solution)
                visible_res = _run_candidate_visible_tests(
                    sandbox,
                    cfg,
                    include_public=(cfg.mode == "D_val"),
                )
                code_candidates_tested_count += 1
                failures = _visible_failure_count(visible_res)
                visible_passed = visible_res.returncode == 0
                if visible_passed:
                    code_candidates_visible_pass_count += 1
                code_candidate_records.append(
                    {
                        "source": source,
                        "solution": candidate_solution,
                        "visible_passed": visible_passed,
                        "visible_failures": failures,
                        "visible_returncode": visible_res.returncode,
                    }
                )

            consider_code_candidate(initial_solution, "initial")
            for candidate_index in range(2, cfg.code_candidates + 1):
                strategy = REPAIR_STRATEGIES[
                    (candidate_index - 2) % len(REPAIR_STRATEGIES)
                ]
                candidate_messages = [
                    {"role": "system", "content": _system_with_skill(code_skill)},
                    {
                        "role": "user",
                        "content": dual_code_candidate_prompt(
                            prompt_md,
                            starter,
                            candidate_index,
                            strategy,
                        ),
                    },
                ]
                try:
                    candidate_reply = code_client.chat(candidate_messages)
                except Exception:
                    code_candidate_model_error_count += 1
                    break
                code_tokens_in += candidate_reply.tokens_in
                code_tokens_out += candidate_reply.tokens_out
                record.tokens_in += candidate_reply.tokens_in
                record.tokens_out += candidate_reply.tokens_out
                candidate_files = extract_files(candidate_reply.text)
                candidate_solution = candidate_files.get("solution.py")
                if candidate_solution is not None:
                    consider_code_candidate(
                        candidate_solution,
                        f"candidate_{candidate_index}",
                    )

            if code_candidate_records:
                passing_candidates = [
                    candidate for candidate in code_candidate_records
                    if candidate["visible_passed"]
                ]
                selected_candidate = (
                    passing_candidates[0]
                    if passing_candidates
                    else min(
                        code_candidate_records,
                        key=lambda candidate: (
                            candidate["visible_failures"],
                            candidate["visible_returncode"],
                        ),
                    )
                )
                sandbox.write("solution.py", selected_candidate["solution"])
                code_candidate_best_failures = int(selected_candidate["visible_failures"])
                code_candidates_selected_visible_pass = bool(selected_candidate["visible_passed"])
                if selected_candidate["source"] != "initial":
                    code_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The harness selected this independent initial code candidate "
                                "after running visible tests. Use it as the current solution "
                                "for future repair steps."
                            ),
                        }
                    )
                    code_messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "```python\n# solution.py\n"
                                + selected_candidate["solution"].strip()
                                + "\n```"
                            ),
                        }
                    )

        for step in range(cfg.k):
            iters = step + 1
            public_res: Optional[RunResult] = None
            self_res: Optional[RunResult] = None

            if cfg.mode == "D_val":
                has_public = (sandbox.tmp / "public_tests.py").exists()
                if has_public and sandbox.bash_calls < cfg.max_bash_calls:
                    public_res = sandbox.run_pytest("public_tests.py", cfg.pytest_timeout)
                    last_public = public_res
                    ran_public_tests = True
                elif not has_public:
                    public_res = RunResult(2, "", "public_tests.py is missing for D_val.", False)
                    last_public = public_res
                else:
                    public_res = RunResult(124, "", "bash call budget exhausted before public_tests.py", False)
                    last_public = public_res

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
            if stop_future_repairs_after_model_error:
                break

            solution = sandbox.read("solution.py")
            self_tests = sandbox.read("self_tests.py")
            use_rewrite_repair = previous_repair_applied
            previous_repair_applied = False
            repair_diagnosis = None
            if cfg.mode == "D_val" and validator_client is not None:
                try:
                    diagnosis_reply = validator_client.chat(
                        [
                            {
                                "role": "system",
                                "content": "You are a concise Python debugging assistant.",
                            },
                            {
                                "role": "user",
                                "content": repair_diagnosis_prompt(
                                    prompt_md,
                                    solution,
                                    self_res,
                                    self_tests,
                                    public_res=public_res,
                                ),
                            },
                        ]
                    )
                    repair_diagnosis = _compact_repair_diagnosis(diagnosis_reply.text)
                    repair_diagnosis_count += 1
                    repair_diagnosis_tokens_in += diagnosis_reply.tokens_in
                    repair_diagnosis_tokens_out += diagnosis_reply.tokens_out
                    validator_tokens_in += diagnosis_reply.tokens_in
                    validator_tokens_out += diagnosis_reply.tokens_out
                    record.tokens_in += diagnosis_reply.tokens_in
                    record.tokens_out += diagnosis_reply.tokens_out
                except Exception:
                    repair_diagnosis = None
            if use_rewrite_repair:
                repair_prompt = rewrite_code_repair_prompt(
                    prompt_md,
                    solution,
                    self_res,
                    self_tests,
                    diagnosis=repair_diagnosis,
                    public_res=public_res,
                )
                code_repair_rewrite_count += 1
            else:
                repair_prompt = (
                    validated_code_repair_prompt(
                        prompt_md,
                        solution,
                        self_res,
                        self_tests,
                        diagnosis=repair_diagnosis,
                        public_res=public_res,
                    )
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
                repair_reply = repair_client.chat(code_messages)
            except Exception as e:
                repair_error = _model_error(e, "repair")
                repair_model_error_count += 1
                repair_model_errors.append(repair_error)
                if _is_timeout_error(e):
                    repair_model_timeout_count += 1
                repair_reply = None
                if repair_client is not code_client:
                    try:
                        repair_reply = code_client.chat(code_messages)
                        repair_fallback_to_code_count += 1
                    except Exception as fallback_e:
                        fallback_error = _model_error(fallback_e, "repair_fallback")
                        repair_fallback_to_code_error_count += 1
                        repair_model_errors.append(fallback_error)
                        if _is_timeout_error(fallback_e):
                            repair_fallback_to_code_timeout_count += 1
                if repair_reply is None:
                    repair_aborted_after_model_error = True
                    repair_preserved_solution_count += 1
                    stop_future_repairs_after_model_error = cfg.abort_repair_on_model_error
                    sandbox.write("solution.py", solution)
                    break
            code_tokens_in += repair_reply.tokens_in
            code_tokens_out += repair_reply.tokens_out
            repair_tokens_in += repair_reply.tokens_in
            repair_tokens_out += repair_reply.tokens_out
            record.tokens_in += repair_reply.tokens_in
            record.tokens_out += repair_reply.tokens_out
            code_messages.append({"role": "assistant", "content": repair_reply.text})

            include_public_for_repair = cfg.mode == "D_val"
            max_repair_candidates = max(1, cfg.repair_candidates)
            repair_candidate_search_count += 1
            repair_candidates_requested_total += max_repair_candidates
            candidate_records: list[dict] = []

            def consider_repair_candidate(reply_text: str, source: str) -> None:
                nonlocal code_repair_unchanged_count
                nonlocal repair_candidates_generated_count
                nonlocal repair_candidates_tested_count
                nonlocal repair_candidates_visible_pass_count

                files = extract_files(reply_text)
                candidate_solution = files.get("solution.py")
                if not candidate_solution:
                    return
                repair_candidates_generated_count += 1
                if candidate_solution.strip() == solution.strip():
                    code_repair_unchanged_count += 1
                    return

                sandbox.write("solution.py", candidate_solution)
                visible_res = _run_candidate_visible_tests(
                    sandbox,
                    cfg,
                    include_public=include_public_for_repair,
                )
                repair_candidates_tested_count += 1
                failures = _visible_failure_count(visible_res)
                visible_passed = visible_res.returncode == 0
                if visible_passed:
                    repair_candidates_visible_pass_count += 1
                candidate_records.append(
                    {
                        "source": source,
                        "solution": candidate_solution,
                        "visible_passed": visible_passed,
                        "visible_failures": failures,
                        "visible_returncode": visible_res.returncode,
                    }
                )

            consider_repair_candidate(repair_reply.text, "primary")
            for candidate_index in range(2, max_repair_candidates + 1):
                strategy = REPAIR_STRATEGIES[
                    (candidate_index - 2) % len(REPAIR_STRATEGIES)
                ]
                retry_messages = [
                    {"role": "system", "content": _system_with_skill(code_skill)},
                    {
                        "role": "user",
                        "content": candidate_code_repair_prompt(
                            prompt_md,
                            solution,
                            self_res,
                            self_tests,
                            candidate_index=candidate_index,
                            strategy=strategy,
                            diagnosis=repair_diagnosis,
                            public_res=public_res,
                        ),
                    },
                ]
                try:
                    repair_retry_reply = repair_client.chat(retry_messages)
                except Exception as e:
                    repair_error = _model_error(e, "repair")
                    repair_candidate_model_error_count += 1
                    repair_model_error_count += 1
                    repair_model_errors.append(repair_error)
                    if _is_timeout_error(e):
                        repair_model_timeout_count += 1
                    repair_retry_reply = None
                    if repair_client is not code_client:
                        try:
                            repair_retry_reply = code_client.chat(retry_messages)
                            repair_fallback_to_code_count += 1
                        except Exception as fallback_e:
                            fallback_error = _model_error(fallback_e, "repair_fallback")
                            repair_fallback_to_code_error_count += 1
                            repair_model_errors.append(fallback_error)
                            if _is_timeout_error(fallback_e):
                                repair_fallback_to_code_timeout_count += 1
                    if repair_retry_reply is None:
                        repair_aborted_after_model_error = True
                        stop_future_repairs_after_model_error = cfg.abort_repair_on_model_error
                        break
                code_repair_retry_count += 1
                code_tokens_in += repair_retry_reply.tokens_in
                code_tokens_out += repair_retry_reply.tokens_out
                repair_tokens_in += repair_retry_reply.tokens_in
                repair_tokens_out += repair_retry_reply.tokens_out
                record.tokens_in += repair_retry_reply.tokens_in
                record.tokens_out += repair_retry_reply.tokens_out
                consider_repair_candidate(
                    repair_retry_reply.text,
                    f"candidate_{candidate_index}",
                )

            if candidate_records:
                passing_candidates = [
                    candidate for candidate in candidate_records
                    if candidate["visible_passed"]
                ]
                selected_candidate = (
                    passing_candidates[0]
                    if passing_candidates
                    else min(
                        candidate_records,
                        key=lambda candidate: (
                            candidate["visible_failures"],
                            candidate["visible_returncode"],
                        ),
                    )
                )
                sandbox.write("solution.py", selected_candidate["solution"])
                previous_repair_applied = True
                repair_candidate_best_failures.append(
                    int(selected_candidate["visible_failures"])
                )
                if selected_candidate["visible_passed"]:
                    repair_candidates_selected_visible_pass_count += 1
                if selected_candidate["source"] != "primary":
                    code_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The harness selected this independently generated repair "
                                "candidate after running visible tests. Use it as the "
                                "current solution for future repair steps."
                            ),
                        }
                    )
                    code_messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "```python\n# solution.py\n"
                                + selected_candidate["solution"].strip()
                                + "\n```"
                            ),
                        }
                    )
            else:
                sandbox.write("solution.py", solution)
                code_repair_stagnation_exhausted_count += 1
                if repair_aborted_after_model_error:
                    repair_preserved_solution_count += 1

            if stop_future_repairs_after_model_error and not candidate_records:
                break

        record.extra["code_tokens_in"] = code_tokens_in
        record.extra["code_tokens_out"] = code_tokens_out
        record.extra["test_tokens_in"] = test_tokens_in
        record.extra["test_tokens_out"] = test_tokens_out
        record.extra["repair_tokens_in"] = repair_tokens_in
        record.extra["repair_tokens_out"] = repair_tokens_out
        record.extra["code_repair_retry_count"] = code_repair_retry_count
        record.extra["code_repair_unchanged_count"] = code_repair_unchanged_count
        record.extra["code_repair_stagnation_exhausted_count"] = code_repair_stagnation_exhausted_count
        record.extra["code_repair_rewrite_count"] = code_repair_rewrite_count
        record.extra["repair_candidate_search_count"] = repair_candidate_search_count
        record.extra["repair_candidates_requested_total"] = repair_candidates_requested_total
        record.extra["repair_candidates_generated_count"] = repair_candidates_generated_count
        record.extra["repair_candidates_tested_count"] = repair_candidates_tested_count
        record.extra["repair_candidates_visible_pass_count"] = repair_candidates_visible_pass_count
        record.extra["repair_candidates_selected_visible_pass_count"] = repair_candidates_selected_visible_pass_count
        record.extra["repair_candidate_best_failures"] = repair_candidate_best_failures
        record.extra["repair_candidate_model_error_count"] = repair_candidate_model_error_count
        record.extra["repair_model_error_count"] = repair_model_error_count
        record.extra["repair_model_timeout_count"] = repair_model_timeout_count
        record.extra["repair_fallback_to_code_count"] = repair_fallback_to_code_count
        record.extra["repair_fallback_to_code_error_count"] = repair_fallback_to_code_error_count
        record.extra["repair_fallback_to_code_timeout_count"] = repair_fallback_to_code_timeout_count
        record.extra["repair_aborted_after_model_error"] = repair_aborted_after_model_error
        record.extra["repair_preserved_solution_count"] = repair_preserved_solution_count
        record.extra["repair_model_errors"] = repair_model_errors[:5]
        record.extra["code_candidates_requested_total"] = code_candidates_requested_total
        record.extra["code_candidates_generated_count"] = code_candidates_generated_count
        record.extra["code_candidates_tested_count"] = code_candidates_tested_count
        record.extra["code_candidates_visible_pass_count"] = code_candidates_visible_pass_count
        record.extra["code_candidates_selected_visible_pass"] = code_candidates_selected_visible_pass
        record.extra["code_candidate_best_failures"] = code_candidate_best_failures
        record.extra["code_candidate_model_error_count"] = code_candidate_model_error_count
        record.extra["repair_diagnosis_count"] = repair_diagnosis_count
        record.extra["repair_diagnosis_tokens_in"] = repair_diagnosis_tokens_in
        record.extra["repair_diagnosis_tokens_out"] = repair_diagnosis_tokens_out
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
    repair_client: Optional[OllamaClient] = None,
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
        return _run_task_dual(
            client,
            test_client,
            task_dir,
            cfg,
            record,
            validator_client,
            repair_client,
        )
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
                sandbox.write("self_tests.py", _expand_self_test_assertions(files["self_tests.py"]))
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
