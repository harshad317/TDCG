import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from harness.repo_agent import (
    RepoAgentConfig,
    apply_model_edits,
    build_repo_map,
    build_verifier,
    changed_files,
    extract_file_blocks,
    git_diff,
    load_verification_commands,
    prepare_workspace,
    repo_map_markdown,
    run_repo_agent,
)
from harness.models import ChatResult


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    write(repo / "README.md", "# Demo\n")
    write(repo / "app.py", "def add(a, b):\n    return a + b\n")
    write(repo / "tests" / "test_app.py", "from app import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    write(repo / "node_modules" / "ignored.js", "ignored\n")
    return repo


def test_build_repo_map_identifies_important_and_test_files(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)

    repo_map = build_repo_map(repo)
    markdown = repo_map_markdown(repo_map)

    assert "README.md" in repo_map.important_files
    assert "tests/test_app.py" in repo_map.test_files
    assert not any("node_modules" in item for item in repo_map.tree)
    assert ".py" in repo_map.language_counts
    assert "Repository tree:" in markdown


def test_extract_and_apply_multi_file_edits(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    workspace = prepare_workspace(repo, tmp_path / "artifacts")
    response = """```python
# path: app.py
def add(a, b):
    return a + b + 1
```

```python
# path: tests/test_app.py
from app import add

def test_add():
    assert add(1, 2) == 4
```
"""

    blocks = extract_file_blocks(response)
    changed = apply_model_edits(workspace, response)

    assert sorted(blocks) == ["app.py", "tests/test_app.py"]
    assert changed == ["app.py", "tests/test_app.py"]
    assert changed_files(workspace) == ["app.py", "tests/test_app.py"]
    diff = git_diff(workspace)
    assert "return a + b + 1" in diff


def test_new_files_appear_in_diff_and_changed_list(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    workspace = prepare_workspace(repo, tmp_path / "artifacts")
    response = """```python
# path: pkg/newmod.py
def helper():
    return 42
```
"""

    changed = apply_model_edits(workspace, response)

    assert changed == ["pkg/newmod.py"]
    # Untracked additions must be visible to the final patch/changed-file list.
    assert changed_files(workspace) == ["pkg/newmod.py"]
    diff = git_diff(workspace)
    assert "pkg/newmod.py" in diff
    assert "return 42" in diff


def test_verification_artifacts_excluded_from_diff(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    workspace = prepare_workspace(repo, tmp_path / "artifacts")

    # Model edits a real source file...
    apply_model_edits(
        workspace,
        "```python\n# path: app.py\ndef add(a, b):\n    return a + b + 1\n```\n",
    )
    # ...while verification leaves byproducts behind.
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\x00junk")
    (workspace / ".pytest_cache").mkdir()
    (workspace / ".pytest_cache" / "CACHEDIR.TAG").write_text("junk\n")
    (workspace / ".coverage").write_text("junk\n")

    changed = changed_files(workspace)

    assert changed == ["app.py"]
    diff = git_diff(workspace)
    assert "__pycache__" not in diff
    assert ".pytest_cache" not in diff
    assert ".coverage" not in diff


def test_build_verifier_defaults_to_host(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    workspace = prepare_workspace(repo, tmp_path / "artifacts")
    cfg = RepoAgentConfig(task_dir=tmp_path / "task", repo_path=repo, model="dummy")

    runner, cleanup = build_verifier(cfg, workspace)
    result = runner("echo hello", 10)
    cleanup()

    assert result.returncode == 0
    assert "hello" in result.stdout


def test_build_verifier_docker_requires_image(tmp_path: Path) -> None:
    cfg = RepoAgentConfig(
        task_dir=tmp_path / "task",
        repo_path=tmp_path / "repo",
        model="dummy",
        use_docker=True,
    )
    with pytest.raises(ValueError):
        build_verifier(cfg, tmp_path)


def test_build_verifier_docker_mounts_workspace_without_owning_it(tmp_path: Path, monkeypatch) -> None:
    repo = make_repo(tmp_path)
    workspace = prepare_workspace(repo, tmp_path / "artifacts")
    cfg = RepoAgentConfig(
        task_dir=tmp_path / "task",
        repo_path=repo,
        model="dummy",
        use_docker=True,
        docker_image="alpine:3",
    )

    monkeypatch.setattr("harness.docker_sandbox.ensure_image", lambda *a, **k: True)

    runner, cleanup = build_verifier(cfg, workspace)
    cleanup()  # must NOT delete the mounted workspace

    assert workspace.exists()
    assert (workspace / "app.py").exists()


def test_load_verification_commands_dedupes_sources(tmp_path: Path) -> None:
    task = tmp_path / "task"
    write(task / "prompt.md", "Fix the repo.\n")
    write(task / "checks.sh", "python -m pytest\n")
    write(
        task / "repo_agent.json",
        json.dumps({"verification_commands": ["python -m pytest", "ruff check ."]}),
    )

    commands = load_verification_commands(task, ["python -m pytest"])

    assert commands == ["python -m pytest", "ruff check .", "bash checks.sh"]


def test_dry_run_writes_repo_agent_artifacts(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    task = tmp_path / "task"
    artifact_dir = tmp_path / "agent_artifacts"
    log_path = tmp_path / "runs.jsonl"
    write(task / "prompt.md", "# Fix add\n\nMake the add behavior correct.\n")
    write(task / "checks.sh", "python -m pytest\n")

    result = run_repo_agent(
        RepoAgentConfig(
            task_dir=task,
            repo_path=repo,
            model="dummy",
            artifact_dir=artifact_dir,
            log_path=log_path,
            dry_run=True,
        )
    )

    assert result.passed is None
    assert (artifact_dir / "repo_map.md").exists()
    assert (artifact_dir / "workspace" / "app.py").exists()
    assert (artifact_dir / "workspace" / "checks.sh").exists()
    row = json.loads(log_path.read_text().strip())
    assert row["mode"] == "D_val_agent"
    assert row["extra"]["workflow"]["repo_map"] is True


def test_full_repo_agent_workflow_with_fake_model(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    write(repo / "README.md", "# Demo\n")
    write(repo / "app.py", "def add(a, b):\n    return a - b\n")
    write(
        repo / "tests" / "test_app.py",
        "from app import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
    )
    task = tmp_path / "task"
    artifact_dir = tmp_path / "agent_artifacts"
    write(task / "prompt.md", "# Fix add\n\nMake `add` return the sum.\n")

    class FakeClient:
        def chat(self, messages):
            system = messages[0]["content"]
            if "planning model" in system:
                return ChatResult(
                    text=json.dumps(
                        {
                            "summary": "Fix add implementation.",
                            "files_to_inspect": ["app.py", "tests/test_app.py"],
                            "change_plan": ["Update app.py.", "Run tests."],
                            "verification_commands": [f"{sys.executable} -m pytest -q"],
                            "risk_notes": [],
                        }
                    ),
                    tokens_in=10,
                    tokens_out=20,
                )
            if "code-writing model" in system:
                return ChatResult(
                    text="""```python
# path: app.py
def add(a, b):
    return a + b
```
""",
                    tokens_in=30,
                    tokens_out=40,
                )
            if "code-review model" in system:
                return ChatResult(
                    text="Verdict: APPROVE\n\nFindings:\n- No blocking issues.\n",
                    tokens_in=50,
                    tokens_out=60,
                )
            raise AssertionError(system)

    monkeypatch.setattr("harness.repo_agent.build_client", lambda *args, **kwargs: FakeClient())

    result = run_repo_agent(
        RepoAgentConfig(
            task_dir=task,
            repo_path=repo,
            model="fake",
            artifact_dir=artifact_dir,
            log_path=tmp_path / "runs.jsonl",
        )
    )

    assert result.passed is True
    assert result.changed_files == ["app.py"]
    assert result.final_error_type is None
    assert "return a + b" in (artifact_dir / "prediction.diff").read_text()
    assert "Verdict: APPROVE" in (artifact_dir / "review.md").read_text()
    assert "PASS:" in (artifact_dir / "pr.md").read_text()
