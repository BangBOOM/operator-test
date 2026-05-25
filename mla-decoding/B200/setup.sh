#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
FLASHINFER_DIR="${FLASHINFER_DIR:-${SCRIPT_DIR}/flashinfer}"
FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
FLASHINFER_BRANCH="${FLASHINFER_BRANCH:-main}"
LOCAL_ATTENTION="${LOCAL_ATTENTION:-${SCRIPT_DIR}/attention.py}"
FLASHINFER_ATTENTION="${FLASHINFER_DIR}/benchmarks/routines/attention.py"

if [[ -d "${FLASHINFER_DIR}/.git" ]]; then
  printf 'Updating FlashInfer at %s\n' "${FLASHINFER_DIR}"
  if [[ -f "${FLASHINFER_ATTENTION}" ]]; then
    git -C "${FLASHINFER_DIR}" checkout -- benchmarks/routines/attention.py
  fi
  git -C "${FLASHINFER_DIR}" fetch origin "${FLASHINFER_BRANCH}" --depth 1
  git -C "${FLASHINFER_DIR}" checkout "${FLASHINFER_BRANCH}"
  git -C "${FLASHINFER_DIR}" pull --ff-only origin "${FLASHINFER_BRANCH}"
else
  if [[ -e "${FLASHINFER_DIR}" ]]; then
    printf 'ERROR: %s exists but is not a git checkout.\n' "${FLASHINFER_DIR}" >&2
    exit 1
  fi

  printf 'Cloning FlashInfer into %s\n' "${FLASHINFER_DIR}"
  git clone --depth 1 --branch "${FLASHINFER_BRANCH}" "${FLASHINFER_REPO}" "${FLASHINFER_DIR}"
fi

if [[ ! -f "${FLASHINFER_DIR}/benchmarks/flashinfer_benchmark.py" ]]; then
  printf 'ERROR: expected benchmark file not found: %s\n' \
    "${FLASHINFER_DIR}/benchmarks/flashinfer_benchmark.py" >&2
  exit 1
fi

if [[ ! -f "${LOCAL_ATTENTION}" ]]; then
  printf 'ERROR: local attention override not found: %s\n' "${LOCAL_ATTENTION}" >&2
  exit 1
fi

if [[ ! -d "$(dirname -- "${FLASHINFER_ATTENTION}")" ]]; then
  printf 'ERROR: expected FlashInfer routines directory not found: %s\n' \
    "$(dirname -- "${FLASHINFER_ATTENTION}")" >&2
  exit 1
fi

install -m 0644 "${LOCAL_ATTENTION}" "${FLASHINFER_ATTENTION}"

mkdir -p "${SCRIPT_DIR}/results"

printf 'FlashInfer is ready: %s\n' "${FLASHINFER_DIR}"
printf 'Installed attention override: %s -> %s\n' "${LOCAL_ATTENTION}" "${FLASHINFER_ATTENTION}"
printf 'Run a single benchmark with: %s/run_mla_single.sh\n' "${SCRIPT_DIR}"
printf 'Run batch benchmarks with: %s/run_mla_bench.sh\n' "${SCRIPT_DIR}"
