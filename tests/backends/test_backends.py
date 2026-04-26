"""Backend tests: docker cmd rendering, native subprocess execution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.backends import (
    JobSpec,
    LocalDockerBackend,
    LocalNativeBackend,
    Mount,
)


# ── LocalDockerBackend: rendering ─────────────────────────────────────


def test_docker_renders_minimal_command() -> None:
    spec = JobSpec(image="img:tag", verb="train")
    cmd = LocalDockerBackend().render_command(spec)
    assert cmd == [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--shm-size=16g",
        "img:tag", "train",
    ]


def test_docker_renders_mounts_env_args() -> None:
    spec = JobSpec(
        image="img:tag",
        verb="train",
        args=["--lr", "1e-4"],
        mounts=[
            Mount(host="/ws", target="/workspace"),
            Mount(host="/data", target="/data", readonly=True),
        ],
        env={"HF_TOKEN": "tok", "AEVOLVE_CYCLE": "5"},
    )
    cmd = LocalDockerBackend().render_command(spec)
    assert "-v" in cmd
    assert "/ws:/workspace" in cmd
    assert "/data:/data:ro" in cmd
    # env passed as -e KEY=VAL; order is dict-insertion order in py3.7+.
    assert cmd.count("-e") == 2
    assert "HF_TOKEN=tok" in cmd
    assert "AEVOLVE_CYCLE=5" in cmd
    # verb precedes args, both after image.
    idx_image = cmd.index("img:tag")
    assert cmd[idx_image + 1] == "train"
    assert cmd[idx_image + 2:] == ["--lr", "1e-4"]


def test_docker_respects_no_gpus_and_no_shm() -> None:
    spec = JobSpec(image="img:tag", verb="infer", gpus=None, shm_size=None)
    cmd = LocalDockerBackend().render_command(spec)
    assert "--gpus" not in cmd
    assert not any(c.startswith("--shm-size") for c in cmd)


def test_docker_custom_gpus_and_workdir() -> None:
    spec = JobSpec(image="img:tag", verb="t", gpus="0,1", workdir="/workspace")
    cmd = LocalDockerBackend().render_command(spec)
    assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "0,1"
    assert "--workdir" in cmd and cmd[cmd.index("--workdir") + 1] == "/workspace"


def test_docker_custom_binary() -> None:
    spec = JobSpec(image="img", verb="v")
    cmd = LocalDockerBackend(docker_bin="podman").render_command(spec)
    assert cmd[0] == "podman"


# ── LocalNativeBackend ─────────────────────────────────────────────────


def test_native_requires_script() -> None:
    spec = JobSpec(image="__native__", verb="train")
    with pytest.raises(ValueError, match="native_script"):
        LocalNativeBackend().render_command(spec)


def test_native_renders_python_invocation(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n")
    spec = JobSpec(
        image="__native__",
        verb="train",
        args=["--cfg", "x.yaml"],
        native_script=script,
    )
    cmd = LocalNativeBackend(python_bin=sys.executable).render_command(spec)
    assert cmd == [sys.executable, str(script), "--cfg", "x.yaml"]


def test_native_submit_runs_script_and_writes_log(tmp_path: Path) -> None:
    script = tmp_path / "hello.py"
    script.write_text(
        "import os, sys\n"
        "print('HELLO', os.environ.get('AEVOLVE_CYCLE', '?'))\n"
        "sys.stderr.write('WARN\\n')\n"
    )
    log = tmp_path / "run.log"
    spec = JobSpec(
        image="__native__",
        verb="train",
        env={"AEVOLVE_CYCLE": "7"},
        log_path=log,
        native_script=script,
    )
    rc = LocalNativeBackend(python_bin=sys.executable).submit(spec)
    assert rc == 0
    body = log.read_text()
    assert "HELLO 7" in body
    assert "WARN" in body  # stderr merged into log


def test_native_submit_raises_on_nonzero(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("import sys; sys.exit(3)\n")
    spec = JobSpec(image="__native__", verb="v", native_script=script)
    with pytest.raises(subprocess.CalledProcessError):
        LocalNativeBackend(python_bin=sys.executable).submit(spec)


def test_native_submit_no_check_returns_exit_code(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("import sys; sys.exit(3)\n")
    spec = JobSpec(image="__native__", verb="v", native_script=script)
    rc = LocalNativeBackend(python_bin=sys.executable).submit(spec, check=False)
    assert rc == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
