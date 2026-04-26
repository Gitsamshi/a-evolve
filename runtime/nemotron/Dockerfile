# aevolve/nemotron-runtime — Tier 1 base image for the CoEvolve nemo stack.
#
# This image contains ONLY the framework: CUDA, torch, tinker-cookbook, and a
# thin dispatcher entrypoint. Workspace code (recipes, envs, cycle_plan.yaml,
# prompts) is bind-mounted in at run time and is NEVER baked in. Rebuild
# cadence: dep upgrades only.
#
# Build:   ./runtime/nemotron/build.sh            # tag: aevolve/nemotron-runtime:YYYY-MM-DD
# Usage:   docker run --rm --gpus all \
#              -v $(pwd)/seed_workspaces/nemo_reason:/workspace \
#              -v $PWD/evolution_workdir/_data:/data:ro \
#              -v $PWD/out:/out \
#              aevolve/nemotron-runtime:<tag> runner --recipe sft
#
# See .claude/a-evolve-plan_v2.md Part 2.E for the three-tier split
# (runtime / backends / workspace) and Part 3 PR 4 for the rebuild rationale.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="9.0;10.0" \
    AEVOLVE_RUNTIME=nemotron

# ── 1. System deps ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        git ca-certificates build-essential ninja-build curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python -m pip install --upgrade pip setuptools wheel

# ── 2. Torch (cu128 wheels) ─────────────────────────────────────────
# Pinned per c640a01 parity. Bump in lock-step with tinker-cookbook's
# supported torch range when upgrading.
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# ── 3. Core ML deps ─────────────────────────────────────────────────
# transformers/peft/accelerate versions track what tinker-cookbook ships.
# Pins here keep the image reproducible across rebuilds.
RUN pip install \
        transformers==4.47.1 \
        peft==0.19.1 \
        "accelerate>=1.13,<1.14" \
        datasets==4.8.4 \
        sentencepiece==0.2.1 \
        pyyaml==6.0.3 \
        pandas \
        numpy

# ── 4. FlashAttention (matched to torch) ────────────────────────────
# --no-build-isolation lets it see the installed torch instead of pulling
# a different version during wheel build.
RUN pip install flash-attn --no-build-isolation

# ── 5. tinker-cookbook (recipes, renderers, merge profiles, primitives) ──
# PR5 swaps the trainer primitives onto cookbook APIs. Installing here
# eliminates ~1000 LOC of custom code the feat-branch shipped
# (mamba_ssm shim, DP fan-out driver, Nemotron merge profile).
#
# TODO(PR5): Pin once the local_client shim (Open Q1) lands. For now we
# intentionally leave the dep commented — the image still builds; the
# runner layer in PR5 will add the pip install.
# RUN pip install tinker-cookbook

# ── 6. ptxas Blackwell fix (carried over from feat-branch) ──────────
# CUDA 12.8's bundled ptxas mis-assembles certain PTX on sm_100; replace
# with the fixed binary from the 12.8.2 toolkit sub-release. Script is a
# no-op when the bundled version is already correct.
COPY ptxas-blackwell-fix.sh /opt/aevolve/ptxas-blackwell-fix.sh
RUN bash /opt/aevolve/ptxas-blackwell-fix.sh

# ── 7. Entrypoint ───────────────────────────────────────────────────
# Dispatches verb → script. Currently knows two verbs (train, inference);
# PR5 adds the ``runner`` verb that reads cycle_plan.yaml.
COPY entrypoint.py /opt/aevolve/entrypoint.py
RUN chmod +x /opt/aevolve/entrypoint.py

WORKDIR /workspace

ENTRYPOINT ["python", "/opt/aevolve/entrypoint.py"]
