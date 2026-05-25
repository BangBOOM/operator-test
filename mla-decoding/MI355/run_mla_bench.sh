#!/usr/bin/env bash
set -euo pipefail

AITER_DIR="${AITER_DIR:-/sgl-workspace/aiter}"
PROFILE_DIR="${PROFILE_DIR:-/home/weyang/log/mla_traces}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BENCHMARK="${BENCHMARK:-${SCRIPT_DIR}/test_mla_persistent_prof.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${BATCH_SIZES:-}" ]]; then
  read -r -a BATCH_SIZE_LIST <<< "${BATCH_SIZES}"
else
  BATCH_SIZE_LIST=(1 2 4 8 12 16 24 32 48 64 96 128 192 256 512)
fi

if [[ -n "${CONFIGS:-}" ]]; then
  read -r -a CONFIG_LIST <<< "${CONFIGS}"
else
  CONFIG_LIST=("${TP:-1}:${QLEN:-1}")
fi

DTYPE="${DTYPE:-fp8}"
KV_DTYPE="${KV_DTYPE:-fp8}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
TOTAL_QO_HEADS="${TOTAL_QO_HEADS:-128}"
K_DIM="${K_DIM:-512}"
QN_DIM="${QN_DIM:-512}"
QR_DIM="${QR_DIM:-64}"
VH_DIM="${VH_DIM:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-1}"
MAX_SEQLEN="${MAX_SEQLEN:-32}"
PROFILE_WARMUP="${PROFILE_WARMUP:-3}"
PROFILE_ITERS="${PROFILE_ITERS:-10}"
GREP_PATTERN="${GREP_PATTERN:-mla_a8w8_.*}"

if [[ ! -d "${AITER_DIR}" ]]; then
  printf 'ERROR: AITER_DIR does not exist: %s\n' "${AITER_DIR}" >&2
  printf 'Expected MI355 aiter directory inside the container: /sgl-workspace/aiter\n' >&2
  exit 1
fi

if [[ ! -f "${BENCHMARK}" ]]; then
  printf 'ERROR: benchmark file not found: %s\n' "${BENCHMARK}" >&2
  printf 'Run setup first: ./setup.sh\n' >&2
  exit 1
fi

mkdir -p "${PROFILE_DIR}"

make_n_arg() {
  local tp="$1"
  local qlen="$2"

  if [[ -n "${N_ARG:-}" ]]; then
    printf '%s\n' "${N_ARG}"
    return
  fi

  if ! [[ "${tp}" =~ ^[0-9]+$ && "${tp}" -gt 0 ]]; then
    printf 'ERROR: TP must be a positive integer, got: %s\n' "${tp}" >&2
    return 1
  fi

  if ! [[ "${qlen}" =~ ^[0-9]+$ && "${qlen}" -gt 0 ]]; then
    printf 'ERROR: QLEN must be a positive integer, got: %s\n' "${qlen}" >&2
    return 1
  fi

  if (( TOTAL_QO_HEADS % tp != 0 )); then
    printf 'ERROR: TOTAL_QO_HEADS=%s is not divisible by TP=%s\n' "${TOTAL_QO_HEADS}" "${tp}" >&2
    return 1
  fi

  printf '%s,%s\n' "$((TOTAL_QO_HEADS / tp))" "${qlen}"
}

for config in "${CONFIG_LIST[@]}"; do
  IFS=':' read -r tp qlen <<< "${config}"
  if [[ -z "${tp}" || -z "${qlen}" ]]; then
    printf 'ERROR: invalid CONFIGS item: %s. Expected format TP:QLEN, for example 8:4\n' "${config}" >&2
    exit 1
  fi

  n_arg="$(make_n_arg "${tp}" "${qlen}")"

  for batch_size in "${BATCH_SIZE_LIST[@]}"; do
    printf '=== TP: %s QLEN: %s N_ARG: %s Batch Size: %s ===\n' \
      "${tp}" "${qlen}" "${n_arg}" "${batch_size}"

    output="$(
      "${PYTHON_BIN}" "${BENCHMARK}" \
        -d "${DTYPE}" -kvd "${KV_DTYPE}" \
        -c "${CONTEXT_LEN}" \
        -b "${batch_size}" \
        -n "${n_arg}" \
        -k "${K_DIM}" -qn "${QN_DIM}" -qr "${QR_DIM}" -vh "${VH_DIM}" \
        -blk "${BLOCK_SIZE}" \
        -ms "${MAX_SEQLEN}" \
        --torch-profile "${PROFILE_DIR}" \
        --torch-profile-warmup "${PROFILE_WARMUP}" \
        --torch-profile-iters "${PROFILE_ITERS}" 2>&1
    )" || {
      printf 'ERROR: benchmark failed for TP=%s QLEN=%s batch_size=%s\n' \
        "${tp}" "${qlen}" "${batch_size}" >&2
      printf '%s\n' "${output}" >&2
      exit 1
    }

    matched="$(printf '%s\n' "${output}" | grep -E "${GREP_PATTERN}" || true)"
    if [[ -n "${matched}" ]]; then
      printf '%s\n' "${matched}"
    else
      printf 'WARN: no output matched GREP_PATTERN=%s\n' "${GREP_PATTERN}" >&2
      printf '%s\n' "${output}" >&2
    fi

    printf '\n'
  done
done
