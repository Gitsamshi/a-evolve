"""BaseTrainer + Backend integration tests.

Verifies that BaseTrainer builds correct JobSpecs from a manifest and
delegates to the injected backend — no real docker / training required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.agents.trainers import BaseTrainer, HfPeftTrainer
from agent_evolve.backends import Backend, JobSpec, LocalDockerBackend, LocalNativeBackend
from agent_evolve.types import Task


class RecordingBackend(Backend):
    """Swallows submissions, records the JobSpec for assertions."""

    def __init__(self) -> None:
        self.submitted: list[JobSpec] = []

    def render_command(self, spec: JobSpec) -> list[str]:  # pragma: no cover - not used
        return [spec.image, spec.verb, *spec.args]

    def submit(self, spec: JobSpec, *, check: bool = True) -> int:
        self.submitted.append(spec)
        if spec.log_path is not None:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            spec.log_path.write_text(
                "{'loss': 0.5, 'learning_rate': 1e-4, 'epoch': 0.1}\n"
                "{'eval_loss': 0.4, 'epoch': 1.0}\n"
            )
        return 0


def _make_workspace(root: Path, docker_image: str = "img:tag") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "system.md").write_text("ok")
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "contract_version": "1.1",
                "agent": {
                    "type": "trainer",
                    "entrypoint": "agent_evolve.agents.trainers.HfPeftTrainer",
                },
                "training": {
                    "docker_image": docker_image,
                    "train_args": ["--lr", "1e-4"],
                    "inference_args": ["--bsz", "8"],
                    "ckpt_path": "adapter",
                },
            },
            sort_keys=False,
        )
    )
    (root / "training_config.yaml").write_text("lr: 1e-4\n")
    # BaseTrainer copies the workspace to the per-cycle workdir; include a
    # stub train.py so local mode has a script to exec.
    (root / "train.py").write_text("print('ok')\n")
    # For inference bind-mount test.
    docker_dir = root / "docker"
    docker_dir.mkdir(exist_ok=True)
    (docker_dir / "inference.py").write_text("print('infer')\n")
    return root


def test_reload_parses_training_block(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path / "ws", docker_image="img:v1")
    t = HfPeftTrainer(ws, work_root=tmp_path / "runs", backend=RecordingBackend())
    assert t.docker_image == "img:v1"
    assert t.train_args == ["--lr", "1e-4"]
    assert t.inference_args == ["--bsz", "8"]
    assert t.ckpt_rel_path == "adapter"


def test_solve_builds_docker_jobspec(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path / "ws", docker_image="img:v1")
    backend = RecordingBackend()
    t = HfPeftTrainer(ws, work_root=tmp_path / "runs", backend=backend)
    task = Task(id="c1", input="")
    traj = t.solve(task)

    assert len(backend.submitted) == 1
    spec = backend.submitted[0]
    assert spec.image == "img:v1"
    assert spec.verb == "train"
    assert spec.args == ["--lr", "1e-4"]
    # Standard mounts: workspace, data, out.
    targets = {m.target for m in spec.mounts}
    assert {"/workspace", "/data", "/out"}.issubset(targets)
    # /data is always read-only.
    data = next(m for m in spec.mounts if m.target == "/data")
    assert data.readonly is True
    # Cycle env baked in.
    assert spec.env["AEVOLVE_CYCLE"] == "c1"
    assert spec.env["AEVOLVE_VERB"] == "train"
    # Log parsed into steps — HfPeft style (dict-per-line).
    assert any(s["type"] == "train_step" for s in traj.steps)
    assert any(s["type"] == "eval" for s in traj.steps)
    # ckpt path threaded through.
    assert traj.output.endswith("adapter")


def test_solve_local_sentinel_uses_native_spec(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path / "ws", docker_image="__local__")
    backend = RecordingBackend()
    t = HfPeftTrainer(ws, work_root=tmp_path / "runs", backend=backend)
    task = Task(id="c2", input="")
    t.solve(task)

    spec = backend.submitted[0]
    assert spec.image == "__local__"
    assert spec.verb == "train"
    assert spec.native_script is not None
    # Native mode passes AEVOLVE_WORKSPACE so the recipe can resolve paths
    # without mounts.
    assert "AEVOLVE_WORKSPACE" in spec.env
    assert spec.gpus is None and spec.shm_size is None


def test_backend_auto_selection(tmp_path: Path) -> None:
    """Without an explicit backend, image sentinel picks docker vs native."""
    ws_docker = _make_workspace(tmp_path / "wsd", docker_image="img:tag")
    ws_local = _make_workspace(tmp_path / "wsl", docker_image="__local__")
    td = HfPeftTrainer(ws_docker, work_root=tmp_path / "runs_d")
    tl = HfPeftTrainer(ws_local, work_root=tmp_path / "runs_l")
    assert isinstance(td._backend(), LocalDockerBackend)
    assert isinstance(tl._backend(), LocalNativeBackend)


def test_solve_errors_when_image_missing(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path / "ws", docker_image="")
    t = HfPeftTrainer(ws, work_root=tmp_path / "runs", backend=RecordingBackend())
    with pytest.raises(RuntimeError, match="docker_image"):
        t.solve(Task(id="c", input=""))


def test_run_inference_builds_jobspec(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path / "ws", docker_image="img:v1")
    backend = RecordingBackend()
    t = HfPeftTrainer(ws, work_root=tmp_path / "runs", backend=backend)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    out = tmp_path / "infer_out"

    t.run_inference(ckpt_dir=ckpt, config_path_in_workspace="eval.yaml", mode="val", out_dir=out)
    spec = backend.submitted[0]
    assert spec.verb == "inference"
    assert "--config" in spec.args and "eval.yaml" in spec.args
    # Inference-time mounts include /ckpt:ro and /workspace:ro.
    targets = {(m.target, m.readonly) for m in spec.mounts}
    assert ("/ckpt", True) in targets
    assert ("/workspace", True) in targets
    # The workspace's docker/inference.py bind-mount overrides the baked copy.
    assert any(m.target == "/opt/aevolve/inference.py" for m in spec.mounts)
    assert spec.env["AEVOLVE_INFER_MODE"] == "val"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
