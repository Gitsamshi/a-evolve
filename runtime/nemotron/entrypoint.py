"""Image dispatcher for aevolve/nemotron-runtime.

Translates the positional docker argv into a concrete python invocation:

  - ``train``      → ``python /workspace/train.py <rest>``
  - ``inference``  → ``python /workspace/docker/inference.py <rest>``
                     (or ``/opt/aevolve/inference.py`` if the workspace
                     doesn't ship one — kept for backwards compat with
                     v1.1 workspaces)
  - ``runner``     → ``python -m agent_evolve.training.runner <rest>``
                     (PR5 — cycle_plan.yaml dispatcher; entry point is
                     already reserved so workspaces can opt-in before
                     PR5 ships the real module)

Exit code: forwarded from the subprocess. stdout/stderr stream directly
to the caller's log sink.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _die(msg: str, code: int = 2) -> None:
    print(f"[entrypoint] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def _exec(cmd: list[str]) -> int:
    print(f"[entrypoint] exec → {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def run_train(extra: list[str]) -> int:
    os.environ.setdefault("AEVOLVE_VERB", "train")
    driver = "/workspace/train.py"
    if not os.path.exists(driver):
        _die(f"missing {driver} (workspace must include train.py for v1.1 recipes)")
    return _exec(["python", driver, *extra])


def run_inference(extra: list[str]) -> int:
    os.environ.setdefault("AEVOLVE_VERB", "inference")
    # Prefer the workspace's driver (lets us iterate without rebuild);
    # fall back to the /opt baked copy once PR5 lands one.
    ws_driver = "/workspace/docker/inference.py"
    opt_driver = "/opt/aevolve/inference.py"
    driver = ws_driver if os.path.exists(ws_driver) else opt_driver
    if not os.path.exists(driver):
        _die(f"no inference driver at {ws_driver} or {opt_driver}")
    return _exec(["python", driver, *extra])


def run_runner(extra: list[str]) -> int:
    os.environ.setdefault("AEVOLVE_VERB", "runner")
    # PR5 ships ``agent_evolve.training.runner`` and pip-installs the
    # agent_evolve package into the image. Until then this verb errors
    # out clearly rather than silently running nothing.
    return _exec(["python", "-m", "agent_evolve.training.runner", *extra])


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        _die("missing verb; expected one of: train | inference | runner")

    verb, rest = argv[0], argv[1:]
    if verb == "train":
        sys.exit(run_train(rest))
    if verb == "inference":
        sys.exit(run_inference(rest))
    if verb == "runner":
        sys.exit(run_runner(rest))
    _die(f"unknown verb {verb!r}; expected one of: train | inference | runner")


if __name__ == "__main__":
    main()
