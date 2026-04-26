"""Execute a :class:`JobSpec` via ``docker run`` on the local host.

Ported from ``BaseTrainer.solve`` / ``run_inference`` (c640a01). The
rendered argv mirrors the feat-branch behaviour so existing images
keep working without changes.
"""

from __future__ import annotations

import logging
import subprocess

from .base import Backend, JobSpec

logger = logging.getLogger(__name__)


class LocalDockerBackend(Backend):
    """Run jobs with ``docker run --rm`` against the host daemon."""

    def __init__(self, docker_bin: str = "docker"):
        self.docker_bin = docker_bin

    def render_command(self, spec: JobSpec) -> list[str]:
        cmd: list[str] = [self.docker_bin, "run", "--rm"]
        if spec.gpus:
            cmd += ["--gpus", spec.gpus]
        if spec.shm_size:
            cmd += [f"--shm-size={spec.shm_size}"]
        for m in spec.mounts:
            cmd += ["-v", m.as_docker_arg()]
        if spec.workdir:
            cmd += ["--workdir", spec.workdir]
        for k, v in spec.env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd.append(spec.image)
        cmd.append(spec.verb)
        cmd.extend(spec.args)
        return cmd

    def submit(self, spec: JobSpec, *, check: bool = True) -> int:
        cmd = self.render_command(spec)
        logger.info("docker submit: %s", " ".join(cmd))
        if spec.log_path is not None:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(spec.log_path, "w") as log_file:
                result = subprocess.run(
                    cmd, check=check, stdout=log_file, stderr=subprocess.STDOUT
                )
        else:
            result = subprocess.run(cmd, check=check)
        return result.returncode
