#!/usr/bin/env bash
# Replace CUDA 12.8's bundled ptxas with the 12.8.2 sub-release version on
# Blackwell (sm_100) hosts. Bundled 12.8.0/12.8.1 ptxas mis-assembles certain
# PTX sequences emitted by torch 2.10's inductor — manifests as wrong-number
# kernels (silent) or build failures from Triton (loud).
#
# Idempotent: detects the already-fixed binary and exits 0 without touching it.
# No-op on non-Blackwell builds (harmless).
#
# Carried over from feat-branch (c640a01). Remove when the image moves to
# CUDA >= 12.8.2 as the base tag.

set -euo pipefail

PTXAS_BIN="${PTXAS_BIN:-/usr/local/cuda/bin/ptxas}"
FIXED_URL="${PTXAS_FIX_URL:-}"  # leave unset in-tree; CI sets it at build time

if [[ ! -x "${PTXAS_BIN}" ]]; then
    echo "[ptxas-fix] no ptxas at ${PTXAS_BIN}; skipping (no-op)."
    exit 0
fi

version="$("${PTXAS_BIN}" --version 2>/dev/null | awk '/release/ {print $NF}' || true)"
echo "[ptxas-fix] current ptxas: ${version:-unknown}"

# The fixed blob ships as 12.8.2+; anything >= that is already good.
case "${version}" in
    V12.8.[2-9]*|V12.[9-9]*|V1[3-9].*)
        echo "[ptxas-fix] version ${version} already fixed; no action."
        exit 0
        ;;
esac

if [[ -z "${FIXED_URL}" ]]; then
    echo "[ptxas-fix] PTXAS_FIX_URL unset; skipping swap (best-effort)."
    exit 0
fi

tmp="$(mktemp)"
echo "[ptxas-fix] downloading fixed ptxas from ${FIXED_URL}"
curl -fsSL -o "${tmp}" "${FIXED_URL}"
chmod +x "${tmp}"
mv "${tmp}" "${PTXAS_BIN}"
echo "[ptxas-fix] replaced ${PTXAS_BIN} (now $("${PTXAS_BIN}" --version | awk '/release/ {print $NF}'))"
