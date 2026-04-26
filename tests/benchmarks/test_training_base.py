"""TrainingBenchmarkAdapter contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.benchmarks import BenchmarkAdapter, TrainingBenchmarkAdapter
from agent_evolve.types import Feedback, Task, Trajectory


class FakeTrainingBenchmark(TrainingBenchmarkAdapter):
    """Minimal concrete adapter used to exercise the contract."""

    def __init__(self) -> None:
        self.evaluate_batch_calls: list[tuple[list[Task], Trajectory, object]] = []

    def get_tasks(self, split: str = "train", limit: int = 10) -> list[Task]:
        return [Task(id=f"t{i}", input=f"q{i}") for i in range(min(2, limit))]

    def evaluate_batch(self, tasks, trajectory, trainer):
        self.evaluate_batch_calls.append((tasks, trajectory, trainer))
        return [Feedback(success=True, score=1.0, detail=f"ok {t.id}") for t in tasks]


def test_is_a_benchmark_adapter() -> None:
    assert issubclass(TrainingBenchmarkAdapter, BenchmarkAdapter)


def test_evaluate_raises_with_batch_hint() -> None:
    bench = FakeTrainingBenchmark()
    task = Task(id="t0", input="q")
    traj = Trajectory(task_id="t0", output="/tmp/ckpt")
    with pytest.raises(NotImplementedError, match="evaluate_batch"):
        bench.evaluate(task, traj)


def test_evaluate_batch_is_called_with_trainer_and_tasks() -> None:
    bench = FakeTrainingBenchmark()
    tasks = bench.get_tasks(limit=2)
    traj = Trajectory(task_id="cycle-1", output="/tmp/ckpt")
    trainer_sentinel = object()  # engine would pass the real trainer instance
    feedbacks = bench.evaluate_batch(tasks, traj, trainer_sentinel)
    assert len(feedbacks) == 2
    assert all(f.success and f.score == 1.0 for f in feedbacks)
    # The adapter received the exact objects the engine handed it.
    (got_tasks, got_traj, got_trainer), = bench.evaluate_batch_calls
    assert got_tasks == tasks
    assert got_traj is traj
    assert got_trainer is trainer_sentinel


def test_abstract_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        TrainingBenchmarkAdapter()  # type: ignore[abstract]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
