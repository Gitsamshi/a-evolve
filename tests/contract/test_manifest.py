"""Manifest v1.2 round-trip + backward-compat tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.contract.manifest import (
    CURRENT_CONTRACT_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    Manifest,
)
from agent_evolve.contract.schema import validate_workspace


def _write(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_current_version_is_1_2() -> None:
    assert CURRENT_CONTRACT_VERSION == "1.2"
    assert SUPPORTED_CONTRACT_VERSIONS == frozenset({"1.0", "1.1", "1.2"})


def test_v10_backcompat_no_training_no_procedures(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "manifest.yaml",
        {
            "name": "legacy",
            "version": "0.1.0",
            "contract_version": "1.0",
            "agent": {"type": "reference", "entrypoint": "pkg.mod.Agent"},
            "evolvable_layers": ["prompts", "skills", "memory"],
            "reload_strategy": "hot",
        },
    )
    m = Manifest.from_yaml(p)
    assert m.contract_version == "1.0"
    assert m.training is None
    assert m.procedures is None
    # to_dict omits the optional blocks when unset
    out = m.to_dict()
    assert "training" not in out
    assert "procedures" not in out


def test_v11_training_block_parsed(tmp_path: Path) -> None:
    training = {
        "docker_image": "aevolve/nemotron-runtime:2026-04-26",
        "train_args": ["--lr", "1e-4"],
        "inference_args": [],
        "ckpt_path": "adapter",
    }
    p = _write(
        tmp_path / "manifest.yaml",
        {
            "name": "trainer-ws",
            "version": "0.1.0",
            "contract_version": "1.1",
            "agent": {"type": "trainer", "entrypoint": "pkg.trainers.HfPeftTrainer"},
            "training": training,
        },
    )
    m = Manifest.from_yaml(p)
    assert m.contract_version == "1.1"
    assert m.agent_type == "trainer"
    assert m.training == training
    assert m.procedures is None


def test_v12_procedures_block_parsed(tmp_path: Path) -> None:
    procedures = {
        "recipes": {
            "sft": "recipes.sft:run",
            "sdft": "recipes.sdft:run",
        },
        "primitives": "agent_evolve.training.primitives",
        "evolvability": {
            "cycle_plan.yaml": "evolvable",
            "envs/": "evolvable",
            "prompts/": "evolvable",
        },
    }
    p = _write(
        tmp_path / "manifest.yaml",
        {
            "name": "nemo_reason",
            "version": "0.1.0",
            "contract_version": "1.2",
            "agent": {"type": "trainer", "entrypoint": "pkg.trainers.HfPeftTrainer"},
            "training": {"docker_image": "aevolve/nemotron-runtime:2026-04-26"},
            "procedures": procedures,
        },
    )
    m = Manifest.from_yaml(p)
    assert m.contract_version == "1.2"
    assert m.procedures == procedures


def test_roundtrip_preserves_all_blocks(tmp_path: Path) -> None:
    m = Manifest(
        name="nemo_reason",
        version="0.2.0",
        contract_version="1.2",
        entrypoint="pkg.trainers.HfPeftTrainer",
        agent_type="trainer",
        evolvable_layers=["prompts", "memory"],
        reload_strategy="cold",
        training={"docker_image": "img:tag", "ckpt_path": "adapter"},
        procedures={"recipes": {"sft": "recipes.sft:run"}},
    )
    path = tmp_path / "manifest.yaml"
    m.save(path)
    loaded = Manifest.from_yaml(path)
    assert loaded == m


def test_defaults_to_current_version_when_missing(tmp_path: Path) -> None:
    # If contract_version is omitted, Manifest should default to CURRENT.
    p = _write(tmp_path / "manifest.yaml", {"name": "fresh", "agent": {"entrypoint": "x.Y"}})
    m = Manifest.from_yaml(p)
    assert m.contract_version == CURRENT_CONTRACT_VERSION


def _seed_workspace(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "system.md").write_text("you are helpful")
    _write(root / "manifest.yaml", manifest)


def test_validate_workspace_accepts_1_0(tmp_path: Path) -> None:
    _seed_workspace(
        tmp_path / "ws",
        {"name": "legacy", "contract_version": "1.0", "agent": {"entrypoint": "x.Y"}},
    )
    assert validate_workspace(tmp_path / "ws") == []


def test_validate_workspace_accepts_1_1(tmp_path: Path) -> None:
    _seed_workspace(
        tmp_path / "ws",
        {
            "name": "trainer",
            "contract_version": "1.1",
            "agent": {"type": "trainer", "entrypoint": "x.Y"},
            "training": {"docker_image": "img:tag"},
        },
    )
    assert validate_workspace(tmp_path / "ws") == []


def test_validate_workspace_accepts_1_2(tmp_path: Path) -> None:
    _seed_workspace(
        tmp_path / "ws",
        {
            "name": "proc",
            "contract_version": "1.2",
            "agent": {"type": "trainer", "entrypoint": "x.Y"},
            "procedures": {"recipes": {"sft": "recipes.sft:run"}},
        },
    )
    assert validate_workspace(tmp_path / "ws") == []


def test_validate_workspace_rejects_unknown_version(tmp_path: Path) -> None:
    _seed_workspace(
        tmp_path / "ws",
        {"name": "future", "contract_version": "2.0", "agent": {"entrypoint": "x.Y"}},
    )
    errs = validate_workspace(tmp_path / "ws")
    assert errs, "expected a version-mismatch error"
    assert any("Contract version mismatch" in e for e in errs)
    assert any("2.0" in e for e in errs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
