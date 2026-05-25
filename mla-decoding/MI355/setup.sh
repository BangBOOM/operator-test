#!/usr/bin/env bash
set -euo pipefail

AITER_DIR="${AITER_DIR:-/sgl-workspace/aiter}"
PROFILE_DIR="${PROFILE_DIR:-/home/weyang/log/mla_traces}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BENCHMARK="${BENCHMARK:-${SCRIPT_DIR}/test_mla_persistent_prof.py}"

if [[ ! -d "${AITER_DIR}" ]]; then
  printf 'ERROR: AITER_DIR does not exist: %s\n' "${AITER_DIR}" >&2
  printf 'Expected MI355 benchmark directory inside the container: /sgl-workspace/aiter\n' >&2
  exit 1
fi

if [[ ! -f "${BENCHMARK}" ]]; then
  printf 'ERROR: benchmark file not found: %s\n' "${BENCHMARK}" >&2
  exit 1
fi

mkdir -p "${PROFILE_DIR}"

printf 'MI355 MLA benchmark environment is ready.\n'
printf 'AITER_DIR=%s\n' "${AITER_DIR}"
printf 'BENCHMARK=%s\n' "${BENCHMARK}"
printf 'PROFILE_DIR=%s\n' "${PROFILE_DIR}"
printf 'Run batch benchmarks with: ./run_mla_bench.sh\n'
