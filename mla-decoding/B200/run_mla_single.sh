#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
FLASHINFER_DIR="${FLASHINFER_DIR:-${SCRIPT_DIR}/flashinfer}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

BACKEND="${BACKEND:-trtllm-native}"
PAGE_SIZE="${PAGE_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
S_QO="${S_QO:-4}"
S_KV="${S_KV:-8192}"
NUM_QO_HEADS="${NUM_QO_HEADS:-16}"
NUM_KV_HEADS="${NUM_KV_HEADS:-1}"
HEAD_DIM_CKV="${HEAD_DIM_CKV:-512}"
HEAD_DIM_KPE="${HEAD_DIM_KPE:-64}"
Q_DTYPE="${Q_DTYPE:-fp8_e4m3}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
NUM_ITERS="${NUM_ITERS:-100}"
DRY_RUN_ITERS="${DRY_RUN_ITERS:-10}"
PROFILE="${PROFILE:-0}"
NSYS_OUTPUT="${NSYS_OUTPUT:-${RESULTS_DIR}/trtllm_mla_decode}"

if [[ ! -f "${FLASHINFER_DIR}/benchmarks/flashinfer_benchmark.py" ]]; then
  printf 'ERROR: FlashInfer benchmark not found at %s\n' \
    "${FLASHINFER_DIR}/benchmarks/flashinfer_benchmark.py" >&2
  printf 'Run setup first: %s/setup.sh\n' "${SCRIPT_DIR}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"
cd "${FLASHINFER_DIR}"

cmd=(
  python benchmarks/flashinfer_benchmark.py
  --routine BatchMLAPagedAttentionWrapper
  --backends "${BACKEND}"
  --page_size "${PAGE_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --s_qo "${S_QO}"
  --s_kv "${S_KV}"
  --num_qo_heads "${NUM_QO_HEADS}"
  --num_kv_heads "${NUM_KV_HEADS}"
  --head_dim_ckv "${HEAD_DIM_CKV}"
  --head_dim_kpe "${HEAD_DIM_KPE}"
  --q_dtype "${Q_DTYPE}"
  --kv_dtype "${KV_DTYPE}"
  --num_iters "${NUM_ITERS}"
  --dry_run_iters "${DRY_RUN_ITERS}"
  -vv
)

if [[ "${PROFILE}" == "1" ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    printf 'ERROR: PROFILE=1 requires nsys, but nsys was not found in PATH.\n' >&2
    exit 1
  fi

  exec nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
    -o "${NSYS_OUTPUT}" \
    "${cmd[@]}"
fi

exec "${cmd[@]}"
