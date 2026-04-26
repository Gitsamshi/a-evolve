"""BaseTrainer -- evolvable agent that runs training/inference via a :class:`Backend`.

The trainer holds no execution logic. It:
  1. copies the current workspace into a per-cycle workdir,
  2. builds a :class:`JobSpec` from the manifest's ``training:`` block,
  3. delegates the actual run to a :class:`Backend` instance, and
  4. parses the resulting log into ``Trajectory.steps``.

The sentinel ``docker_image: __local__`` selects :class:`LocalNativeBackend`;
any other image tag selects :class:`LocalDockerBackend`. See
``.claude/a-evolve-plan_v2.md`` Part 1 for the refactor rationale.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from ...backends import Backend, JobSpec, LocalDockerBackend, LocalNativeBackend, Mount
from ...contract.manifest import Manifest
from ...protocol.base_agent import BaseAgent
from ...types import Task, Trajectory

logger = logging.getLogger(__name__)

LOCAL_IMAGE_SENTINEL = "__local__"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


class BaseTrainer(BaseAgent):
    """Docker-run-or-native trainer. Subclasses override ``_summarize_log``."""

    def __init__(
        self,
        workspace_dir: str | Path,
        work_root: str | Path | None = None,
        backend: Backend | None = None,
    ):
        workspace_path = Path(workspace_dir).resolve()
        # Sibling of the workspace. Nested dirs would be picked up by the
        # evolution git repo and re-copied into later cycles' workdirs.
        default_root = workspace_path.parent / f"{workspace_path.name}_runs"
        self.work_root: Path = Path(work_root).resolve() if work_root else default_root

        # Populated by reload_from_fs().
        self.training_config: dict = {}
        self.data_config: dict = {}
        self.base_ckpt: dict = {}
        self.docker_image: str = ""
        self.train_args: list[str] = []
        self.inference_args: list[str] = []
        self.ckpt_rel_path: str = ""
        self.manifest_training: dict = {}

        # Injected backend wins. Otherwise pick based on image sentinel at
        # submit time (the image is not known until reload_from_fs runs).
        self._explicit_backend = backend

        super().__init__(workspace_dir)

    # ── File System Contract ─────────────────────────────────────────

    def reload_from_fs(self) -> None:
        super().reload_from_fs()
        root = self.workspace.root

        self.training_config = _load_yaml(root / "training_config.yaml")
        self.data_config = _load_yaml(root / "data_config.yaml")
        self.base_ckpt = _load_yaml(root / "base_ckpt.yaml")

        manifest_path = root / "manifest.yaml"
        if manifest_path.exists():
            m = Manifest.from_yaml(manifest_path)
            t = m.training or {}
            self.manifest_training = t
            self.docker_image = t.get("docker_image", "")
            self.train_args = list(t.get("train_args", []))
            self.inference_args = list(t.get("inference_args", []))
            self.ckpt_rel_path = t.get("ckpt_path", "")

    # ── Backend selection ────────────────────────────────────────────

    @property
    def _is_local(self) -> bool:
        return self.docker_image == LOCAL_IMAGE_SENTINEL

    def _backend(self) -> Backend:
        if self._explicit_backend is not None:
            return self._explicit_backend
        return LocalNativeBackend() if self._is_local else LocalDockerBackend()

    # ── Workdir preparation ──────────────────────────────────────────

    def _prepare_workdir(self, workdir: Path) -> None:
        """Copy the current workspace into ``workdir`` (excluding engine artifacts)."""
        if workdir.exists():
            shutil.rmtree(workdir)
        shutil.copytree(
            self.workspace.root,
            workdir,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns("evolution", ".git", "__pycache__", "*.pyc"),
        )
        (workdir / "out").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _data_dir() -> Path:
        return Path(os.environ.get("AEVOLVE_DATA_DIR", "./evolution_workdir/_data")).resolve()

    @staticmethod
    def _models_dir() -> Path | None:
        """Host path bind-mounted to ``/models`` (local base ckpts).

        Returns None when ``AEVOLVE_MODELS_DIR`` is unset; the container
        can still pull from HF Hub via network.
        """
        val = os.environ.get("AEVOLVE_MODELS_DIR")
        if not val:
            return None
        p = Path(val)
        return p.resolve() if p.exists() else None

    # Env vars forwarded from host into the container when set. Tokens for
    # gated HF models, optional HF cache redirection.
    _FORWARDED_ENV: tuple[str, ...] = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    )

    @classmethod
    def _env_passthrough(cls) -> dict[str, str]:
        return {name: os.environ[name] for name in cls._FORWARDED_ENV if os.environ.get(name)}

    @classmethod
    def _models_mounts(cls) -> list[Mount]:
        mounts: list[Mount] = []
        mdir = cls._models_dir()
        if mdir is not None:
            mounts.append(Mount(host=str(mdir), target="/models", readonly=True))
        shared = os.environ.get("AEVOLVE_SHARED_FS_ROOT")
        if shared:
            sp = Path(shared).resolve()
            if sp.exists():
                mounts.append(Mount(host=str(sp), target=str(sp), readonly=True))
        return mounts

    def _image_driver_mount(self) -> list[Mount]:
        """Bind-mount ``<workspace>/docker/inference.py`` over the baked-in copy.

        Lets us iterate on the inference driver without rebuilding the image.
        """
        driver = self.workspace.root / "docker" / "inference.py"
        if not driver.exists():
            return []
        return [Mount(host=str(driver.resolve()), target="/opt/aevolve/inference.py", readonly=True)]

    # ── Train verb ───────────────────────────────────────────────────

    def solve(self, task: Task) -> Trajectory:
        if not self.docker_image:
            raise RuntimeError(
                "manifest.training.docker_image is not set; cannot invoke train verb."
            )
        if not self.ckpt_rel_path:
            raise RuntimeError(
                "manifest.training.ckpt_path is not set; cannot locate trained ckpt."
            )

        workdir = self.work_root / f"cycle-{task.id}"
        self._prepare_workdir(workdir)
        out_dir = workdir / "out"
        data_dir = self._data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        log_path = workdir / "train.log"

        if self._is_local:
            spec = JobSpec(
                image=LOCAL_IMAGE_SENTINEL,
                verb="train",
                args=[],
                env={
                    "AEVOLVE_WORKSPACE": str(workdir.resolve()),
                    "AEVOLVE_OUT": str(out_dir.resolve()),
                    "AEVOLVE_DATA": str(data_dir),
                    "AEVOLVE_CYCLE": str(task.id),
                    "AEVOLVE_VERB": "train",
                },
                gpus=None,
                shm_size=None,
                log_path=log_path,
                native_script=workdir / "train.py",
            )
        else:
            spec = JobSpec(
                image=self.docker_image,
                verb="train",
                args=list(self.train_args),
                mounts=[
                    Mount(host=str(workdir.resolve()), target="/workspace"),
                    Mount(host=str(data_dir), target="/data", readonly=True),
                    Mount(host=str(out_dir.resolve()), target="/out"),
                    *self._models_mounts(),
                ],
                env={
                    "AEVOLVE_CYCLE": str(task.id),
                    "AEVOLVE_VERB": "train",
                    **self._env_passthrough(),
                },
                log_path=log_path,
            )

        self._backend().submit(spec)
        ckpt_path = out_dir / self.ckpt_rel_path
        return Trajectory(
            task_id=task.id,
            output=str(ckpt_path),
            steps=self._summarize_log(log_path),
        )

    # ── Inference verb ───────────────────────────────────────────────

    def run_inference(
        self,
        ckpt_dir: Path,
        config_path_in_workspace: str,
        mode: str,
        out_dir: Path,
    ) -> Path:
        """Invoke the image's ``inference`` verb. Returns ``out_dir``."""
        if not self.docker_image:
            raise RuntimeError(
                "manifest.training.docker_image is not set; cannot invoke inference verb."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self._data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"inference_{mode}.log"

        if self._is_local:
            driver = self.workspace.root / "docker" / "inference.py"
            if not driver.exists():
                raise FileNotFoundError(f"Local mode requires {driver}")
            spec = JobSpec(
                image=LOCAL_IMAGE_SENTINEL,
                verb="inference",
                args=["--config", config_path_in_workspace],
                env={
                    "AEVOLVE_WORKSPACE": str(self.workspace.root.resolve()),
                    "AEVOLVE_OUT": str(out_dir.resolve()),
                    "AEVOLVE_DATA": str(data_dir),
                    "AEVOLVE_CKPT": str(Path(ckpt_dir).resolve()),
                    "AEVOLVE_VERB": "inference",
                    "AEVOLVE_INFER_MODE": mode,
                },
                gpus=None,
                shm_size=None,
                log_path=log_path,
                native_script=driver,
            )
        else:
            spec = JobSpec(
                image=self.docker_image,
                verb="inference",
                args=["--config", config_path_in_workspace, *self.inference_args],
                mounts=[
                    Mount(host=str(self.workspace.root.resolve()), target="/workspace", readonly=True),
                    Mount(host=str(data_dir), target="/data", readonly=True),
                    Mount(host=str(Path(ckpt_dir).resolve()), target="/ckpt", readonly=True),
                    Mount(host=str(out_dir.resolve()), target="/out"),
                    *self._models_mounts(),
                    *self._image_driver_mount(),
                ],
                env={
                    "AEVOLVE_VERB": "inference",
                    "AEVOLVE_INFER_MODE": mode,
                    **self._env_passthrough(),
                },
                log_path=log_path,
            )

        self._backend().submit(spec)
        return out_dir

    # ── Log parsing (override in subclasses) ─────────────────────────

    _STEP_RE = re.compile(
        r"(?P<key>loss|lr|learning_rate|val_loss|val_bpb|bpb|step|epoch)\s*[:=]\s*"
        r"(?P<value>-?\d+(?:\.\d+)?(?:e-?\d+)?)",
        re.IGNORECASE,
    )

    def _summarize_log(self, log_path: Path, max_entries: int = 50) -> list[dict[str, Any]]:
        """Parse training log into a compact list of step dicts.

        Default implementation picks up lines containing common metrics
        (``loss``, ``lr``, ``step``, ``val_*``). Subclasses override for
        framework-specific formats.
        """
        if not log_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(log_path) as f:
            for line in f:
                matches = list(self._STEP_RE.finditer(line))
                if not matches:
                    continue
                entry: dict[str, Any] = {"type": "train_step"}
                for m in matches:
                    key = m.group("key").lower()
                    try:
                        entry[key] = float(m.group("value"))
                    except ValueError:
                        continue
                if any(k in entry for k in ("val_loss", "val_bpb", "bpb")):
                    entry["type"] = "eval"
                entries.append(entry)
        return entries[-max_entries:]
