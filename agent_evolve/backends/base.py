"""Backend ABC and the :class:`JobSpec` value object it consumes."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mount:
    """A host-path → container-path bind mount.

    Ignored by :class:`LocalNativeBackend`; the trainer is expected to
    pass the same paths via ``env`` so the recipe can resolve them.
    """

    host: str
    target: str
    readonly: bool = False

    def as_docker_arg(self) -> str:
        suffix = ":ro" if self.readonly else ""
        return f"{self.host}:{self.target}{suffix}"


@dataclass
class JobSpec:
    """One verb invocation, independent of execution target.

    A trainer fills this in for each train / inference call; a backend
    turns it into ``docker run`` args, a local subprocess, or a K8s Job.

    Fields:
      image: container image tag, or ``"__native__"`` to force the local
        native backend (the trainer typically picks the backend directly,
        so this is mostly advisory).
      verb: positional entrypoint arg (e.g. ``"train"`` or ``"inference"``).
      args: remaining positional CLI args appended after ``verb``.
      mounts: host↔container binds. Docker applies them; native ignores.
      env: extra env vars injected into the job.
      gpus: ``"all"`` for `--gpus all`, ``None`` for CPU, or a device list
        string (e.g. ``"0,1"``).
      shm_size: ``--shm-size`` value, e.g. ``"16g"``; ``None`` to omit.
      workdir: container workdir (``--workdir``) — optional.
      log_path: file the backend writes combined stdout/stderr to.
      native_script: absolute path to the python script the native
        backend should exec when ``image == "__native__"``. Ignored by
        docker backend.
    """

    image: str
    verb: str
    args: list[str] = field(default_factory=list)
    mounts: list[Mount] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    gpus: str | None = "all"
    shm_size: str | None = "16g"
    workdir: str | None = None
    log_path: Path | None = None
    native_script: Path | None = None


class Backend(ABC):
    """Execute a :class:`JobSpec` and return its exit code."""

    @abstractmethod
    def render_command(self, spec: JobSpec) -> list[str]:
        """Return the argv the backend would exec for ``spec``.

        Side-effect free; used by tests and dry runs.
        """

    @abstractmethod
    def submit(self, spec: JobSpec, *, check: bool = True) -> int:
        """Run ``spec`` to completion.

        When ``check`` is True (the default), a non-zero exit raises
        ``subprocess.CalledProcessError``.
        """
