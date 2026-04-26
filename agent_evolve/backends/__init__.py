"""Execution backends for training/inference verbs.

A ``Backend`` runs a :class:`JobSpec` (a container-agnostic description of
one verb invocation) and returns an exit code. The intent is that
trainers build JobSpecs and backends are interchangeable — local docker
for development, native host execution for __local__ mode, K8s for
fan-out later.
"""

from .base import Backend, JobSpec, Mount
from .local_docker import LocalDockerBackend
from .local_native import LocalNativeBackend

__all__ = [
    "Backend",
    "JobSpec",
    "Mount",
    "LocalDockerBackend",
    "LocalNativeBackend",
]
