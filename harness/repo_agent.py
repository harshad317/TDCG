"""Repo-level D_val agent runner.

This module is the bridge from single-file benchmark tasks to agentic
software-engineering tasks. It works on a real repository checkout, asks the
model to inspect/plan before editing, applies multi-file changes, runs visible
verification commands, performs a review pass, and writes PR-style artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .log import JsonlLogger, RunRecord, now
from .models import ChatResult, OllamaClient, build_client
from .sandbox import RunResult


MODE = "D_val_agent"
DEFAULT_MAX_FILES = 240
DEFAULT_MAX_FILE_BYTES = 24_000
DEFAULT_MAX_CONTEXT_FILES = 18
DEFAULT_COMMAND_TIMEOUT = 120

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".turbo",
    ".cache",
    "target",
    "vendor",
    "results",
    "outputs",
}

IMPORTANT_NAMES = {
    "AGENTS.md",
    "AGENTS.override.md",
    "README.md",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "pnpm-lock.yaml",
    "package-lock.json",
    "yarn.lock",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "tox.ini",
    "pytest.ini",
}

SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".sql",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".ini",
    ".cfg",
    ".html",
    ".css",
}


@dataclass
class RepoMap:
    root: str
    files_considered: int
    files_included: int
    truncated: bool
    tree: list[str]
    important_files: list[str]
    test_files: list[str]
    package_files: list[str]
    language_counts: dict[str, int]


@dataclass
class RepoAgentConfig:
    task_dir: Path
    repo_path: Path
    model: str
    backend: str = "ollama"
    host: str = "http://127.0.0.1:11434"
    test_model: str | None = None
    validator_model: str | None = None
    review_model: str | None = None
    repair_model: str | None = None
    model_timeout: int = 120
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    max_iterations: int = 3
    max_bash_calls: int = 20
    use_docker: bool = False
    docker_image: str | None = None
    docker_network: str = "none"
    docker_platform: str | None = None
    max_files: int = DEFAULT_MAX_FILES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_context_files: int = DEFAULT_MAX_CONTEXT_FILES
    verification_commands: list[str] = field(default_factory=list)
    artifact_dir: Path | None = None
    log_path: Path | None = None
    batch_tag: str = "repo_agent"
    dry_run: bool = False


@dataclass
class RepoAgentResult:
    task_id: str
    passed: bool | None
    iterations_used: int
    bash_calls_used: int
    tokens_in: int
    tokens_out: int
    artifact_dir: str
    changed_files: list[str]
    verification_results: list[dict]
    final_error_type: str | None


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)[:120] or "task"


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:2048]
    except OSError:
        return True
    return b"\0" in chunk


def _candidate_file(path: Path) -> bool:
    if path.name in IMPORTANT_NAMES:
        return True
    return path.suffix.lower() in SOURCE_SUFFIXES


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_repo_map(
    repo_root: Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> RepoMap:
    repo_root = repo_root.resolve()
    tree: list[str] = []
    important: list[str] = []
    tests: list[str] = []
    packages: list[str] = []
    language_counts: dict[str, int] = {}
    considered = 0
    included = 0
    truncated = False

    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
        current = Path(dirpath)
        for filename in sorted(filenames):
            if filename == ".DS_Store":
                continue
            paths.append(current / filename)

    for path in sorted(paths, key=lambda p: relpath(p, repo_root)):
        rel = relpath(path, repo_root)
        considered += 1
        if not _candidate_file(path) or _is_binary(path):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_file_bytes:
            marker = f"{rel} ({size} bytes, skipped: too large)"
            if len(tree) < max_files:
                tree.append(marker)
            continue
        if included >= max_files:
            truncated = True
            continue
        included += 1
        tree.append(f"{rel} ({size} bytes)")
        lower_rel = rel.lower()
        suffix = path.suffix.lower() or "[no_ext]"
        language_counts[suffix] = language_counts.get(suffix, 0) + 1
        if path.name in IMPORTANT_NAMES:
            important.append(rel)
        if (
            lower_rel.startswith("test")
            or "/test" in lower_rel
            or lower_rel.endswith("_test.py")
            or lower_rel.endswith(".test.ts")
            or lower_rel.endswith(".test.tsx")
            or lower_rel.endswith(".spec.ts")
            or lower_rel.endswith(".spec.tsx")
        ):
            tests.append(rel)
        if path.name in {"package.json", "pyproject.toml", "requirements.txt", "go.mod", "Cargo.toml"}:
            packages.append(rel)

    return RepoMap(
        root=str(repo_root),
        files_considered=considered,
        files_included=included,
        truncated=truncated,
        tree=tree,
        important_files=important,
        test_files=tests,
        package_files=packages,
        language_counts=dict(sorted(language_counts.items())),
    )


def repo_map_markdown(repo_map: RepoMap) -> str:
    lines = [
        f"Repo root: {repo_map.root}",
        f"Files considered: {repo_map.files_considered}",
        f"Files included: {repo_map.files_included}",
        f"Truncated: {repo_map.truncated}",
        "",
        "Important files:",
        *[f"- {path}" for path in repo_map.important_files[:40]],
        "",
        "Test files:",
        *[f"- {path}" for path in repo_map.test_files[:80]],
        "",
        "Package/config files:",
        *[f"- {path}" for path in repo_map.package_files[:40]],
        "",
        "Language/file counts:",
        *[f"- {suffix}: {count}" for suffix, count in repo_map.language_counts.items()],
        "",
        "Repository tree:",
        *[f"- {item}" for item in repo_map.tree],
    ]
    return "\n".join(lines).strip() + "\n"


def load_prompt(task_dir: Path) -> str:
    prompt = task_dir / "prompt.md"
    if prompt.exists():
        return prompt.read_text()
    instance = task_dir / "instance.json"
    if instance.exists():
        data = json.loads(instance.read_text())
        text = data.get("problem_statement") or data.get("instruction")
        if not text:
            raise ValueError(
                f"{instance} has no 'problem_statement' or 'instruction' field"
            )
        return str(text)
    raise FileNotFoundError(f"{task_dir} does not contain prompt.md or instance.json")


def load_verification_commands(task_dir: Path, explicit: Iterable[str] = ()) -> list[str]:
    commands = [cmd for cmd in explicit if cmd.strip()]
    manifest = task_dir / "repo_agent.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        for cmd in data.get("verification_commands", []):
            if isinstance(cmd, str) and cmd.strip():
                commands.append(cmd)
    checks = task_dir / "checks.sh"
    if checks.exists():
        commands.append(f"bash {checks.name}")
    return _dedupe(commands)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


# Verification (pytest, coverage, build tools, ...) writes byproducts into the
# workspace. We exclude them via .git/info/exclude — a local, untracked file — so
# `git add -A` never stages them into prediction.diff. Using .git/info/exclude
# instead of a tracked .gitignore keeps the exclusion itself out of the diff.
GIT_EXCLUDE_PATTERNS = "\n".join(
    [
        "__pycache__/",
        "*.py[cod]",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".coverage",
        ".coverage.*",
        "htmlcov/",
        "*.egg-info/",
        ".tox/",
        ".DS_Store",
        "node_modules/",
    ]
) + "\n"


def prepare_workspace(repo_path: Path, artifact_dir: Path) -> Path:
    repo_path = repo_path.resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise FileNotFoundError(f"repo path not found: {repo_path}")
    workspace = artifact_dir / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(
        repo_path,
        workspace,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".mypy_cache"),
    )
    _run_git(["git", "init"], workspace)
    _run_git(["git", "config", "user.name", "TDCG Repo Agent"], workspace)
    _run_git(["git", "config", "user.email", "tdcg-repo-agent@example.invalid"], workspace)
    exclude_path = workspace / ".git" / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    with exclude_path.open("a") as fh:
        fh.write("\n" + GIT_EXCLUDE_PATTERNS)
    _run_git(["git", "add", "-A"], workspace)
    _run_git(["git", "commit", "-m", "baseline"], workspace)
    return workspace


def _run_git(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc


def run_shell(command: str, cwd: Path, timeout: int) -> RunResult:
    """Run a verification command on the host.

    SECURITY: ``command`` may originate from model output (plan
    ``verification_commands``) and is executed via ``bash -lc`` on the host with
    the caller's environment. Only run this against trusted tasks/models. For
    untrusted inputs, route verification through ``harness.docker_sandbox``
    instead of this function.
    """
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return RunResult(proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as e:
        return RunResult(
            124,
            (e.stdout or "") if isinstance(e.stdout, str) else "",
            (e.stderr or "") if isinstance(e.stderr, str) else "",
            True,
        )


def parse_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
        if match:
            stripped = match.group(1)
    else:
        match = re.search(r"(\{.*\})", stripped, flags=re.S)
        if match:
            stripped = match.group(1)
    return json.loads(stripped)


PATH_HEADER_RE = re.compile(r"^\s*(?:#|//|--|<!--)\s*(?:path|file)\s*:\s*(.+?)\s*(?:-->)?\s*$", re.I)


def extract_file_blocks(text: str) -> dict[str, str]:
    """Extract complete-file edits from fenced blocks.

    Accepted header styles inside each block:
      # path: src/foo.py
      // file: src/foo.ts
    """
    files: dict[str, str] = {}
    for match in re.finditer(r"```[A-Za-z0-9_+.-]*\n(.*?)```", text, flags=re.S):
        body = match.group(1)
        lines = body.splitlines()
        if not lines:
            continue
        header = PATH_HEADER_RE.match(lines[0])
        if not header:
            continue
        path = header.group(1).strip().strip("`")
        content = "\n".join(lines[1:])
        if body.endswith("\n"):
            content += "\n"
        files[path] = content
    return files


def extract_unified_diff(text: str) -> str | None:
    fence = re.search(r"```(?:diff|patch)?\s*(diff --git .*?)```", text, flags=re.S)
    if fence:
        return fence.group(1).strip() + "\n"
    idx = text.find("diff --git ")
    if idx >= 0:
        return text[idx:].strip() + "\n"
    return None


def _safe_workspace_path(workspace: Path, path: str) -> Path:
    normalized = Path(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"unsafe model path: {path}")
    target = (workspace / normalized).resolve()
    try:
        target.relative_to(workspace.resolve())
    except ValueError as e:
        raise ValueError(f"unsafe model path: {path}") from e
    return target


def apply_model_edits(workspace: Path, response: str) -> list[str]:
    files = extract_file_blocks(response)
    if files:
        # Resolve/validate every path up front so a single unsafe path cannot
        # leave the workspace half-edited.
        targets = {path: _safe_workspace_path(workspace, path) for path in files}
        changed: list[str] = []
        for path, content in files.items():
            target = targets[path]
            target.parent.mkdir(parents=True, exist_ok=True)
            old = target.read_text() if target.exists() else None
            if old != content:
                target.write_text(content)
                changed.append(path)
        return sorted(changed)

    patch = extract_unified_diff(response)
    if patch:
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=patch,
            cwd=workspace,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git apply failed:\n{proc.stdout}\n{proc.stderr}")
        return changed_files(workspace)

    raise ValueError("model response contained no file blocks or unified diff")


def _stage_all(workspace: Path) -> None:
    # Stage everything (including newly created files) so that diffs capture
    # untracked additions, not just modifications to already-tracked files.
    subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, text=True, check=True)


def changed_files(workspace: Path) -> list[str]:
    _stage_all(workspace)
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def git_diff(workspace: Path) -> str:
    _stage_all(workspace)
    proc = subprocess.run(
        ["git", "diff", "--cached", "--binary"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def read_files_for_context(
    workspace: Path,
    paths: Iterable[str],
    *,
    max_files: int,
    max_file_bytes: int,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in _dedupe(paths):
        if len(out) >= max_files:
            break
        try:
            target = _safe_workspace_path(workspace, path)
        except ValueError:
            continue
        if not target.exists() or not target.is_file() or _is_binary(target):
            continue
        try:
            text = target.read_text()
        except UnicodeDecodeError:
            continue
        if len(text.encode("utf-8")) > max_file_bytes:
            text = text[:max_file_bytes] + "\n...[truncated]\n"
        out[path] = text
    return out


def context_markdown(files: dict[str, str]) -> str:
    chunks: list[str] = []
    for path, content in files.items():
        lang = Path(path).suffix.lstrip(".")
        chunks.append(f"## {path}\n\n```{lang}\n{content}```")
    return "\n\n".join(chunks)


def _chat(client: OllamaClient, messages: list[dict]) -> ChatResult:
    return client.chat(messages)


def plan_prompt(task_prompt: str, repo_map: str) -> list[dict]:
    system = (
        "You are the planning model in a repo-level coding agent. "
        "Read the task and repository map, then select the smallest useful "
        "set of files to inspect before editing. Output only JSON."
    )
    user = f"""
Task:
{task_prompt}

Repository map:
{repo_map}

Return JSON with this schema:
{{
  "summary": "one sentence",
  "files_to_inspect": ["relative/path"],
  "change_plan": ["step 1", "step 2"],
  "verification_commands": ["command to run from repo root"],
  "risk_notes": ["risk"]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def edit_prompt(
    task_prompt: str,
    repo_map: str,
    plan: dict,
    context: str,
    feedback: str | None = None,
) -> list[dict]:
    system = (
        "You are the code-writing model in a repo-level D_val agent. "
        "Make a correct, minimal multi-file codebase change. Preserve style. "
        "Do not output prose. Output only complete changed files in fenced code "
        "blocks. The first line inside each block must be '# path: relative/path'."
    )
    feedback_block = f"\nVerification feedback from previous attempt:\n{feedback}\n" if feedback else ""
    user = f"""
Task:
{task_prompt}

Repository map:
{repo_map}

Plan:
{json.dumps(plan, indent=2)}

Selected file context:
{context}
{feedback_block}

Output every changed file as:
```python
# path: relative/path.py
<full file contents>
```
Use the correct language tag for non-Python files. Include no unchanged files.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def review_prompt(task_prompt: str, plan: dict, diff: str, verification: list[dict]) -> list[dict]:
    system = (
        "You are the validation and code-review model in a repo-level D_val agent. "
        "Review the patch for correctness, regressions, missing tests, and PR readiness. "
        "Be specific and actionable. Do not invent files or hidden test results."
    )
    user = f"""
Task:
{task_prompt}

Plan:
{json.dumps(plan, indent=2)}

Verification results:
{json.dumps(verification, indent=2)}

Patch:
```diff
{diff}
```

Return markdown with:
1. Verdict: APPROVE or REQUEST_CHANGES
2. Findings: ordered by severity with file paths when possible
3. Suggested follow-up tests
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def summarize_pr(task_prompt: str, changed: list[str], verification: list[dict], review: str) -> str:
    passing = [item["command"] for item in verification if item["returncode"] == 0]
    failing = [item["command"] for item in verification if item["returncode"] != 0]
    title = "Repo-level fix"
    first_line = task_prompt.strip().splitlines()[0] if task_prompt.strip() else ""
    if first_line.startswith("#"):
        title = first_line.lstrip("# ").strip()[:72] or title
    summary_lines = [f"- Updated `{path}`" for path in changed]
    if not summary_lines:
        summary_lines = ["- No file changes were produced."]
    verification_lines = [f"- PASS: `{cmd}`" for cmd in passing]
    verification_lines += [f"- FAIL: `{cmd}`" for cmd in failing]
    if not verification_lines:
        verification_lines = ["- No verification commands were configured."]
    body = [
        f"# {title}",
        "",
        "## Summary",
        *summary_lines,
        "",
        "## Verification",
        *verification_lines,
        "",
        "## Review",
        review.strip() or "No review was produced.",
        "",
    ]
    return "\n".join(line for line in body if line != "") + "\n"


def build_verifier(cfg: "RepoAgentConfig", workspace: Path):
    """Return a ``(command, timeout) -> RunResult`` callable for verification.

    Defaults to host execution. When ``cfg.use_docker`` is set, commands run in a
    Docker container with the workspace mounted and (by default) no network, so
    model-generated commands never execute on the host. Returns ``(runner,
    cleanup)``.
    """
    if not cfg.use_docker:
        return (lambda command, timeout: run_shell(command, workspace, timeout)), (lambda: None)

    if not cfg.docker_image:
        raise ValueError("use_docker=True requires docker_image")

    from .docker_sandbox import DockerSandbox, ensure_image

    if not ensure_image(cfg.docker_image):
        raise RuntimeError(f"docker image unavailable: {cfg.docker_image}")

    sandbox = DockerSandbox(
        image=cfg.docker_image,
        workdir="/workspace",
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        platform=cfg.docker_platform,
        network=cfg.docker_network,
        host_dir=workspace,
    )

    def runner(command: str, timeout: int) -> RunResult:
        r = sandbox.run(["bash", "-lc", command], timeout=timeout)
        return RunResult(r.returncode, r.stdout, r.stderr, r.timed_out)

    return runner, sandbox.cleanup


def run_verification(
    commands: list[str],
    workspace: Path,
    timeout: int,
    runner=None,
) -> list[dict]:
    run_cmd = runner or (lambda command, t: run_shell(command, workspace, t))
    results: list[dict] = []
    for command in commands:
        result = run_cmd(command, timeout)
        results.append(
            {
                "command": command,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        )
    return results


def verification_passed(results: list[dict]) -> bool | None:
    if not results:
        return None
    return all(item["returncode"] == 0 for item in results)


def _artifact_dir(cfg: RepoAgentConfig, task_id: str) -> Path:
    if cfg.artifact_dir:
        return cfg.artifact_dir
    return Path("results/artifacts/repo_agent") / _safe_path_part(cfg.batch_tag) / _safe_path_part(task_id)


def run_repo_agent(cfg: RepoAgentConfig) -> RepoAgentResult:
    task_id = cfg.task_dir.name
    start = now()
    artifact_dir = _artifact_dir(cfg, task_id)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    task_prompt = load_prompt(cfg.task_dir)
    commands = load_verification_commands(cfg.task_dir, cfg.verification_commands)
    workspace = prepare_workspace(cfg.repo_path, artifact_dir)
    checks = cfg.task_dir / "checks.sh"
    if checks.exists():
        shutil.copy2(checks, workspace / "checks.sh")
    repo_map = build_repo_map(workspace, max_files=cfg.max_files, max_file_bytes=cfg.max_file_bytes)
    repo_map_md = repo_map_markdown(repo_map)

    (artifact_dir / "prompt.md").write_text(task_prompt)
    (artifact_dir / "repo_map.md").write_text(repo_map_md)
    (artifact_dir / "repo_map.json").write_text(json.dumps(asdict(repo_map), indent=2) + "\n")

    if cfg.dry_run:
        empty = RepoAgentResult(
            task_id=task_id,
            passed=None,
            iterations_used=0,
            bash_calls_used=0,
            tokens_in=0,
            tokens_out=0,
            artifact_dir=str(artifact_dir),
            changed_files=[],
            verification_results=[],
            final_error_type=None,
        )
        _write_record(cfg, empty, start)
        return empty

    code_client = build_client(
        cfg.model,
        backend=cfg.backend,
        host=cfg.host,
        timeout=cfg.model_timeout,
    )
    planner_client = code_client
    review_client = build_client(
        cfg.review_model or cfg.validator_model or cfg.test_model or cfg.model,
        backend=cfg.backend,
        host=cfg.host,
        timeout=cfg.model_timeout,
    )
    repair_client = (
        build_client(
            cfg.repair_model,
            backend=cfg.backend,
            host=cfg.host,
            timeout=cfg.model_timeout,
        )
        if cfg.repair_model
        else code_client
    )

    verifier, verifier_cleanup = build_verifier(cfg, workspace)

    tokens_in = 0
    tokens_out = 0
    bash_calls = 0
    final_error: str | None = None

    try:
        plan_result = _chat(planner_client, plan_prompt(task_prompt, repo_map_md))
        tokens_in += plan_result.tokens_in
        tokens_out += plan_result.tokens_out
        (artifact_dir / "plan.raw.md").write_text(plan_result.text)
        plan = parse_json_object(plan_result.text)
    except Exception as e:
        plan = {
            "summary": "Fallback plan after planner output failed to parse.",
            "files_to_inspect": repo_map.important_files[:6] + repo_map.test_files[:6],
            "change_plan": ["Inspect likely entry points and tests.", "Make minimal changes.", "Run verification."],
            "verification_commands": [],
            "risk_notes": [f"planner_error: {type(e).__name__}: {e}"],
        }
        final_error = "planner_fallback"

    commands = _dedupe(commands + [cmd for cmd in plan.get("verification_commands", []) if isinstance(cmd, str)])
    (artifact_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    inspect_paths = plan.get("files_to_inspect", [])
    if not isinstance(inspect_paths, list):
        inspect_paths = []
    base_context_paths = (
        [str(path) for path in inspect_paths]
        + repo_map.important_files
        + repo_map.test_files[:8]
    )

    def build_context(extra_paths: Iterable[str] = ()) -> str:
        # Edited files are listed first so they survive the max_context_files cap;
        # read_files_for_context reads current workspace contents.
        files = read_files_for_context(
            workspace,
            list(extra_paths) + base_context_paths,
            max_files=cfg.max_context_files,
            max_file_bytes=cfg.max_file_bytes,
        )
        return context_markdown(files)

    context_md = build_context()
    (artifact_dir / "context.md").write_text(context_md)

    feedback: str | None = None
    verification: list[dict] = []
    iterations_used = 0
    for iteration in range(1, cfg.max_iterations + 1):
        edit_client = code_client if iteration == 1 else repair_client
        if iteration > 1:
            # Repair iterations: refresh context from the current workspace so the
            # model sees the files it just edited, not only the verification log.
            context_md = build_context(changed_files(workspace))
        try:
            edit_result = _chat(
                edit_client, edit_prompt(task_prompt, repo_map_md, plan, context_md, feedback)
            )
        except Exception as e:
            final_error = f"model_error:{type(e).__name__}"
            feedback = f"Model call failed: {type(e).__name__}: {e}"
            continue
        tokens_in += edit_result.tokens_in
        tokens_out += edit_result.tokens_out
        (artifact_dir / f"edit_iter_{iteration}.raw.md").write_text(edit_result.text)
        try:
            changed_this_iter = apply_model_edits(workspace, edit_result.text)
        except Exception as e:
            final_error = f"apply_error:{type(e).__name__}"
            feedback = f"Patch application failed: {type(e).__name__}: {e}"
            continue
        if not changed_this_iter and not changed_files(workspace):
            final_error = "no_changes"
            feedback = "No file changes were produced. Produce complete changed files."
            continue
        # Count only iterations that produced an applied edit.
        iterations_used = iteration
        if commands and bash_calls + len(commands) > cfg.max_bash_calls:
            final_error = "bash_budget_exhausted"
            break
        verification = run_verification(commands, workspace, cfg.command_timeout, runner=verifier)
        bash_calls += len(commands)
        (artifact_dir / f"verification_iter_{iteration}.json").write_text(
            json.dumps(verification, indent=2) + "\n"
        )
        passed = verification_passed(verification)
        if passed is True or passed is None:
            final_error = None
            break
        feedback = json.dumps(verification, indent=2)
        final_error = "verification_failed"

    verifier_cleanup()

    diff = git_diff(workspace)
    changed = changed_files(workspace)
    (artifact_dir / "prediction.diff").write_text(diff)

    review_text = ""
    if diff:
        try:
            review_result = _chat(review_client, review_prompt(task_prompt, plan, diff, verification))
            tokens_in += review_result.tokens_in
            tokens_out += review_result.tokens_out
            review_text = review_result.text
        except Exception as e:
            review_text = (
                "Verdict: REQUEST_CHANGES\n\nFindings:\n"
                f"- Review model call failed: {type(e).__name__}: {e}\n"
            )
            final_error = final_error or f"review_error:{type(e).__name__}"
    else:
        review_text = "Verdict: REQUEST_CHANGES\n\nFindings:\n- No patch was produced.\n"
        final_error = final_error or "no_diff"
    (artifact_dir / "review.md").write_text(review_text)
    (artifact_dir / "pr.md").write_text(summarize_pr(task_prompt, changed, verification, review_text))

    result = RepoAgentResult(
        task_id=task_id,
        passed=verification_passed(verification),
        iterations_used=iterations_used,
        bash_calls_used=bash_calls,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        artifact_dir=str(artifact_dir),
        changed_files=changed,
        verification_results=verification,
        final_error_type=final_error,
    )
    (artifact_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    _write_record(cfg, result, start)
    return result


def _write_record(cfg: RepoAgentConfig, result: RepoAgentResult, start: float) -> None:
    if cfg.log_path is None:
        return
    record = RunRecord(
        task_id=result.task_id,
        model=cfg.model,
        mode=MODE,
        k=cfg.max_iterations,
        passed_public=result.passed,
        passed_self=None,
        passed_hidden=None,
        iterations_used=result.iterations_used,
        bash_calls_used=result.bash_calls_used,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        wall_time_s=round(now() - start, 3),
        final_error_type=result.final_error_type,
    )
    record.extra.update(
        {
            "batch_tag": cfg.batch_tag,
            "artifact_dir": result.artifact_dir,
            "repo_path": str(cfg.repo_path),
            "code_model": cfg.model,
            "repair_model": cfg.repair_model or cfg.model,
            "review_model": cfg.review_model or cfg.validator_model or cfg.test_model or cfg.model,
            "changed_files": result.changed_files,
            "verification_results": result.verification_results,
            "verification_sandbox": "docker" if cfg.use_docker else "host",
            "docker_image": cfg.docker_image if cfg.use_docker else None,
            "workflow": {
                "repo_map": True,
                "plan": True,
                "multi_file_edit": True,
                "review": True,
                "pr_summary": True,
            },
        }
    )
    JsonlLogger(cfg.log_path).write(record)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repo-level D_val_agent protocol.")
    parser.add_argument("--task", required=True, type=Path, help="task directory containing prompt.md")
    parser.add_argument("--repo", required=True, type=Path, help="repository checkout to copy and edit")
    parser.add_argument("--model", required=True, help="code model id")
    parser.add_argument("--test-model", default=None)
    parser.add_argument("--validator-model", default=None)
    parser.add_argument("--review-model", default=None)
    parser.add_argument("--repair-model", default=None)
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model-timeout", type=int, default=120)
    parser.add_argument("--command-timeout", type=int, default=DEFAULT_COMMAND_TIMEOUT)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-bash-calls", type=int, default=20)
    parser.add_argument(
        "--docker",
        action="store_true",
        help="run verification commands in a Docker sandbox instead of on the host",
    )
    parser.add_argument("--docker-image", default=None, help="image to use with --docker")
    parser.add_argument(
        "--docker-network",
        default="none",
        help="docker --network for verification (default: none)",
    )
    parser.add_argument(
        "--docker-platform",
        default=None,
        help="docker --platform (e.g. linux/amd64 for x86 images on arm hosts)",
    )
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-context-files", type=int, default=DEFAULT_MAX_CONTEXT_FILES)
    parser.add_argument("--verify", action="append", default=[], help="verification command from repo root")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=Path("results/repo_agent_runs.jsonl"))
    parser.add_argument("--batch-tag", default="repo_agent")
    parser.add_argument("--dry-run", action="store_true", help="build repo map/artifacts without model calls")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = RepoAgentConfig(
        task_dir=args.task,
        repo_path=args.repo,
        model=args.model,
        backend=args.backend,
        host=args.host,
        test_model=args.test_model,
        validator_model=args.validator_model,
        review_model=args.review_model,
        repair_model=args.repair_model,
        model_timeout=args.model_timeout,
        command_timeout=args.command_timeout,
        max_iterations=args.max_iterations,
        max_bash_calls=args.max_bash_calls,
        use_docker=args.docker,
        docker_image=args.docker_image,
        docker_network=args.docker_network,
        docker_platform=args.docker_platform,
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        max_context_files=args.max_context_files,
        verification_commands=args.verify,
        artifact_dir=args.artifact_dir,
        log_path=args.log,
        batch_tag=args.batch_tag,
        dry_run=args.dry_run,
    )
    result = run_repo_agent(cfg)
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.final_error_type is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
