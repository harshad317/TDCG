"""Per-run sandbox: copies task files into a tempdir, runs commands, scores."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


SANDBOX_FILES = ["prompt.md", "solution.py", "public_tests.py"]
DEFAULT_CMD_TIMEOUT = 10


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class Sandbox:
    """Isolated workspace for one task attempt.

    The sandbox dir contains everything the model can see/edit. hidden_tests.py
    is kept outside and only used by the harness for final scoring.
    """

    def __init__(self, task_dir: Path):
        self.task_dir = Path(task_dir)
        self.tmp = Path(tempfile.mkdtemp(prefix="codehyp_"))
        for name in SANDBOX_FILES:
            src = self.task_dir / name
            if src.exists():
                shutil.copy2(src, self.tmp / name)
        self.bash_calls = 0

    def read(self, name: str) -> str:
        return (self.tmp / name).read_text()

    def write(self, name: str, content: str) -> None:
        # Restrict writes to known files inside the sandbox dir.
        safe = (self.tmp / name).resolve()
        try:
            safe.relative_to(self.tmp.resolve())
        except ValueError:
            raise ValueError(f"unsafe write path: {name}")
        safe.write_text(content)

    def run(self, argv: list[str], timeout: int = DEFAULT_CMD_TIMEOUT) -> RunResult:
        self.bash_calls += 1
        try:
            proc = subprocess.run(
                argv,
                cwd=self.tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONINTMAXSTRDIGITS": "1000000",
                    "PYTHONHASHSEED": "0",
                },
            )
            return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
        except subprocess.TimeoutExpired as e:
            return RunResult(
                returncode=124,
                stdout=(e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr=(e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
                timed_out=True,
            )

    def run_pytest(self, test_file: str, timeout: int = DEFAULT_CMD_TIMEOUT) -> RunResult:
        return self.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=short", test_file],
            timeout=timeout,
        )

    def run_pytest_files(self, test_files: list[str], timeout: int = DEFAULT_CMD_TIMEOUT) -> RunResult:
        return self.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=short", *test_files],
            timeout=timeout,
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def score_hidden(task_dir: Path, sandbox: Sandbox, timeout: int = DEFAULT_CMD_TIMEOUT) -> RunResult:
    """Copy hidden_tests.py into a fresh scoring dir alongside the model's solution.

    We do this in a separate dir so the model never sees hidden_tests.py — even
    though by this point the model has stopped, the invariant is worth keeping.
    """
    score_dir = Path(tempfile.mkdtemp(prefix="codehyp_score_"))
    try:
        shutil.copy2(sandbox.tmp / "solution.py", score_dir / "solution.py")
        shutil.copy2(task_dir / "hidden_tests.py", score_dir / "hidden_tests.py")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=short", "hidden_tests.py"],
            cwd=score_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONINTMAXSTRDIGITS": "1000000",
                "PYTHONHASHSEED": "0",
            },
        )
        return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as e:
        return RunResult(
            returncode=124,
            stdout=(e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            stderr=(e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
            timed_out=True,
        )
    finally:
        shutil.rmtree(score_dir, ignore_errors=True)


def score_self_tests_on_reference(
    task_dir: Path,
    sandbox: Sandbox,
    timeout: int = DEFAULT_CMD_TIMEOUT,
) -> RunResult | None:
    """Run generated self_tests.py against a trusted reference solution if present.

    reference_solution.py is never copied into the model-visible sandbox. The
    harness uses it only as a private oracle to reject self-tests with incorrect
    expected values.
    """
    reference = task_dir / "reference_solution.py"
    self_tests = sandbox.tmp / "self_tests.py"
    if not reference.exists() or not self_tests.exists():
        return None

    score_dir = Path(tempfile.mkdtemp(prefix="codehyp_ref_"))
    try:
        shutil.copy2(reference, score_dir / "solution.py")
        shutil.copy2(self_tests, score_dir / "self_tests.py")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=short", "self_tests.py"],
            cwd=score_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONINTMAXSTRDIGITS": "1000000",
                "PYTHONHASHSEED": "0",
            },
        )
        return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as e:
        return RunResult(
            returncode=124,
            stdout=(e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            stderr=(e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
            timed_out=True,
        )
    finally:
        shutil.rmtree(score_dir, ignore_errors=True)
