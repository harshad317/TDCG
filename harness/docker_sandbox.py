"""Docker exec primitive for repo-level / shell benchmarks.

Wraps `docker run` with a mounted workspace dir. Used by SWE-bench / SWT-Bench
/ Terminal-Bench loaders to execute commands in per-task images without
polluting the host environment.

Image lifecycle is NOT managed here — callers pass an image name; pulling /
building is the caller's responsibility (typically via the official runner).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class DockerRunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class DockerSandbox:
    """Run commands in a per-task container with a mounted workspace.

    Workflow:
        sb = DockerSandbox(image="swebench/sweb.eval.x86_64.django__django-12345")
        sb.write("solution.diff", patch_text)
        result = sb.run(["bash", "-lc", "git apply solution.diff && pytest -q"])
    """

    def __init__(
        self,
        image: str,
        workdir: str = "/workspace",
        env: dict[str, str] | None = None,
        platform: str | None = None,  # "linux/amd64" forced for x86 SWE-bench images on M-series
        network: str = "none",
    ):
        self.image = image
        self.workdir = workdir
        self.env = env or {}
        self.platform = platform
        self.network = network
        self.host_dir = Path(tempfile.mkdtemp(prefix="dockersb_"))
        self.container_name = f"codehyp_{uuid.uuid4().hex[:10]}"

    # ---- workspace ops (on host, mirrored into container) ----

    def write(self, name: str, content: str) -> None:
        target = (self.host_dir / name).resolve()
        try:
            target.relative_to(self.host_dir.resolve())
        except ValueError:
            raise ValueError(f"unsafe path: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def read(self, name: str) -> str:
        return (self.host_dir / name).read_text()

    def copy_in(self, src: Path, name: str | None = None) -> None:
        dst = self.host_dir / (name or Path(src).name)
        if Path(src).is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # ---- exec ----

    def run(self, cmd: list[str], timeout: int = 60) -> DockerRunResult:
        docker_cmd = self._docker_run_cmd(cmd)
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return DockerRunResult(proc.returncode, proc.stdout, proc.stderr, False)
        except subprocess.TimeoutExpired as e:
            # Clean up the container; --rm should handle but kill is belt+braces.
            subprocess.run(
                ["docker", "kill", self.container_name],
                capture_output=True,
                text=True,
            )
            return DockerRunResult(
                returncode=124,
                stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
                stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
                timed_out=True,
            )

    def _docker_run_cmd(self, inner: list[str]) -> list[str]:
        argv = [
            "docker", "run", "--rm",
            "--name", self.container_name,
            "-w", self.workdir,
            "-v", f"{self.host_dir}:{self.workdir}",
            f"--network={self.network}",
        ]
        if self.platform:
            argv += ["--platform", self.platform]
        for k, v in self.env.items():
            argv += ["-e", f"{k}={v}"]
        argv.append(self.image)
        argv += inner
        return argv

    def cleanup(self) -> None:
        # Best-effort container kill (if it survived) and host dir removal.
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            text=True,
        )
        shutil.rmtree(self.host_dir, ignore_errors=True)


def ensure_image(image: str, timeout: int = 600) -> bool:
    """Pull `image` if not present locally. Returns True on success."""
    have = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if have.returncode == 0:
        return True
    pull = subprocess.run(
        ["docker", "pull", image],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return pull.returncode == 0
