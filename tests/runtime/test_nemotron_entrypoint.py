"""Smoke tests for the nemotron runtime's entrypoint dispatcher.

The entrypoint is plain Python — no docker required. We load it via
importlib so we don't need it on PYTHONPATH.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "runtime" / "nemotron" / "entrypoint.py"


def _load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("nemotron_entrypoint", ENTRYPOINT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    """Invoke the entrypoint as a subprocess so sys.exit is real."""
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), *argv],
        capture_output=True,
        text=True,
    )


def test_module_loads() -> None:
    mod = _load()
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_train")
    assert hasattr(mod, "run_inference")
    assert hasattr(mod, "run_runner")


def test_missing_verb_errors_with_hint() -> None:
    r = _run()
    assert r.returncode == 2
    assert "missing verb" in r.stderr
    assert "train" in r.stderr and "inference" in r.stderr and "runner" in r.stderr


def test_unknown_verb_errors_with_hint() -> None:
    r = _run("train-fast")
    assert r.returncode == 2
    assert "unknown verb" in r.stderr
    assert "'train-fast'" in r.stderr


def test_train_errors_when_no_workspace_driver() -> None:
    # No /workspace/train.py on the host — expect the clear "missing" error.
    r = _run("train")
    assert r.returncode == 2
    assert "/workspace/train.py" in r.stderr


def test_inference_errors_when_no_driver() -> None:
    # Neither /workspace/docker/inference.py nor /opt/aevolve/inference.py
    # exist on the host — expect the fallback error.
    r = _run("inference")
    assert r.returncode == 2
    assert "inference driver" in r.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
