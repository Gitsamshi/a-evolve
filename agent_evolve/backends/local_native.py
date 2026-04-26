"""Execute a :class:`JobSpec` directly on the host, no container.

Replaces the ``__local__`` sentinel path in the old ``BaseTrainer``. The
trainer fills ``spec.native_script`` with the python file to exec and
``spec.env`` with the ``AEVOLVE_*`` pointers the recipe reads.
"""

from __future__ import annotations

import logging
import os
import subprocess

from .base import Backend, JobSpec

logger = logging.getLogger(__name__)


class LocalNativeBackend(Backend):
    """Run jobs as a host subprocess (``python <script>``)."""

    def __init__(self, python_bin: str | None = None):
        # Default follows the c640a01 convention: respect $AEVOLVE_PYTHON,
        # fall back to ``python``.
        self.python_bin = python_bin or os.environ.get("AEVOLVE_PYTHON", "python")

    def render_command(self, spec: JobSpec) -> list[str]:
        if spec.native_script is None:
            raise ValueError(
                "LocalNativeBackend requires JobSpec.native_script; "
                f"got verb={spec.verb!r} with no script"
            )
        cmd: list[str] = [self.python_bin, str(spec.native_script)]
        cmd.extend(spec.args)
        return cmd

    def submit(self, spec: JobSpec, *, check: bool = True) -> int:
        cmd = self.render_command(spec)
        env = {**os.environ, **spec.env}
        logger.info("native submit: %s", " ".join(cmd))
        if spec.log_path is not None:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(spec.log_path, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    check=check,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=spec.workdir,
                )
        else:
            result = subprocess.run(cmd, check=check, env=env, cwd=spec.workdir)
        return result.returncode
