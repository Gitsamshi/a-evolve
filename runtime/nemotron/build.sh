#!/usr/bin/env bash
# Build the aevolve/nemotron-runtime Tier-1 base image.
#
# Usage: ./runtime/nemotron/build.sh [tag]
# Default tag: today's date (YYYY-MM-DD).
#
# The image is pure framework — no workspace code is baked in. Rebuild
# only when dep pins in the Dockerfile change, not per-cycle.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:-$(date +%Y-%m-%d)}"
IMAGE="aevolve/nemotron-runtime:${TAG}"

echo "Building ${IMAGE} from ${HERE}"
docker build \
    -t "${IMAGE}" \
    -f "${HERE}/Dockerfile" \
    "${HERE}"

echo
echo "Built ${IMAGE}"
echo
echo "Point a workspace at this image via manifest.yaml:"
echo "  training:"
echo "    docker_image: ${IMAGE}"
