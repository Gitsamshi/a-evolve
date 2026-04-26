"""EvolutionLoop tests for the training_score surfacing patch (PR3).

Exercises the 4b hook: when ``engine.manages_own_evaluation`` is True and
the engine returns ``StepResult.metadata['training_score']``, the loop
overwrites the per-task score_history entry with that value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.benchmarks import BenchmarkAdapter
from agent_evolve.config import EvolveConfig
from agent_evolve.engine.base import EvolutionEngine
from agent_evolve.engine.loop import EvolutionLoop
from agent_evolve.protocol.base_agent import BaseAgent
from agent_evolve.types import Feedback, StepResult, Task, Trajectory


# ── Fixtures / fakes ─────────────────────────────────────────────────


class _StubAgent(BaseAgent):
    """Agent that echoes the task's input. No LLM, no side effects."""

    def solve(self, task: Task) -> Trajectory:
        return Trajectory(task_id=task.id, output=f"echo:{task.input}")


class _StubBenchmark(BenchmarkAdapter):
    """Deterministic benchmark; every trajectory scores ``per_task_score``."""

    def __init__(self, per_task_score: float = 0.1, n_tasks: int = 2) -> None:
        self.per_task_score = per_task_score
        self.n_tasks = n_tasks

    def get_tasks(self, split: str = "train", limit: int = 10) -> list[Task]:
        n = min(self.n_tasks, limit)
        return [Task(id=f"t{i}", input=f"q{i}") for i in range(n)]

    def evaluate(self, task: Task, trajectory: Trajectory) -> Feedback:
        return Feedback(success=True, score=self.per_task_score, detail="stub")


class _FakeEngine(EvolutionEngine):
    """Returns a fixed StepResult; toggles manages_own_evaluation via ctor."""

    def __init__(
        self,
        *,
        own_eval: bool,
        training_score: float | None,
    ) -> None:
        self.manages_own_evaluation = own_eval  # shadows class default
        self._training_score = training_score

    def step(self, workspace, observations, history, trial) -> StepResult:  # type: ignore[override]
        meta: dict = {}
        if self._training_score is not None:
            meta["training_score"] = self._training_score
        return StepResult(mutated=False, summary="noop", metadata=meta)


def _seed_workspace(root: Path) -> Path:
    """Minimum workspace the BaseAgent + EvolutionLoop need to boot."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "system.md").write_text("you are helpful")
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "stub",
                "contract_version": "1.0",
                "agent": {"type": "reference", "entrypoint": "x.Y"},
            },
            sort_keys=False,
        )
    )
    return root


def _make_loop(tmp_path: Path, engine: EvolutionEngine, per_task: float = 0.1) -> EvolutionLoop:
    ws = _seed_workspace(tmp_path / "ws")
    agent = _StubAgent(ws)
    bench = _StubBenchmark(per_task_score=per_task, n_tasks=2)
    cfg = EvolveConfig(batch_size=2, max_cycles=1, egl_window=1)
    return EvolutionLoop(agent=agent, benchmark=bench, engine=engine, config=cfg)


# ── Tests ─────────────────────────────────────────────────────────────


def test_default_manages_own_evaluation_is_false() -> None:
    """Subclasses that don't override inherit False — no behavior change."""
    assert EvolutionEngine.manages_own_evaluation is False


def test_score_history_uses_per_task_mean_when_engine_doesnt_manage(tmp_path: Path) -> None:
    engine = _FakeEngine(own_eval=False, training_score=0.9)
    loop = _make_loop(tmp_path, engine, per_task=0.25)
    result = loop.run(cycles=1)
    # Per-task score=0.25 for 2 tasks → mean 0.25. Engine's 0.9 is ignored
    # because manages_own_evaluation is False.
    assert result.score_history == [pytest.approx(0.25)]


def test_score_history_surfaces_training_score_when_engine_manages(tmp_path: Path) -> None:
    engine = _FakeEngine(own_eval=True, training_score=0.8)
    loop = _make_loop(tmp_path, engine, per_task=0.25)
    result = loop.run(cycles=1)
    # Per-task mean would be 0.25, but the engine surfaced a training_score
    # of 0.8 via metadata, and manages_own_evaluation=True, so loop uses it.
    assert result.score_history == [pytest.approx(0.8)]


def test_manages_own_evaluation_without_training_score_falls_back(tmp_path: Path) -> None:
    """If the engine forgot to emit training_score, loop keeps per-task mean."""
    engine = _FakeEngine(own_eval=True, training_score=None)
    loop = _make_loop(tmp_path, engine, per_task=0.25)
    result = loop.run(cycles=1)
    assert result.score_history == [pytest.approx(0.25)]


def test_training_score_must_be_numeric(tmp_path: Path) -> None:
    """Non-numeric training_score is ignored (doesn't crash the loop)."""
    class _BadEngine(EvolutionEngine):
        manages_own_evaluation = True

        def step(self, workspace, observations, history, trial) -> StepResult:  # type: ignore[override]
            return StepResult(mutated=False, summary="x", metadata={"training_score": "0.9"})

    loop = _make_loop(tmp_path, _BadEngine(), per_task=0.25)
    result = loop.run(cycles=1)
    assert result.score_history == [pytest.approx(0.25)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
