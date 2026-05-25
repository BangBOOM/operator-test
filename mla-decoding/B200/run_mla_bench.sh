#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
FLASHINFER_DIR="${FLASHINFER_DIR:-${SCRIPT_DIR}/flashinfer}"

if [[ -n "${BATCH_SIZES:-}" ]]; then
 read -r -a BATCH_SIZE_LIST <<< "${BATCH_SIZES}"
else
 BATCH_SIZE_LIST=(1 2 4 8 12 16 24 32 48 64 96 128 192 256 512)
fi

BACKEND="${BACKEND:-trtllm-native}"
PAGE_SIZE="${PAGE_SIZE:-32}"
S_QO="${S_QO:-1}"
S_KV="${S_KV:-8192}"
NUM_QO_HEADS="${NUM_QO_HEADS:-128}"
NUM_KV_HEADS="${NUM_KV_HEADS:-1}"
HEAD_DIM_CKV="${HEAD_DIM_CKV:-512}"
HEAD_DIM_KPE="${HEAD_DIM_KPE:-64}"
Q_DTYPE="${Q_DTYPE:-fp8_e4m3}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
NUM_ITERS="${NUM_ITERS:-100}"
DRY_RUN_ITERS="${DRY_RUN_ITERS:-10}"

if [[ ! -f "${FLASHINFER_DIR}/benchmarks/flashinfer_benchmark.py" ]]; then
 printf 'ERROR: FlashInfer benchmark not found at %s\n' \
   "${FLASHINFER_DIR}/benchmarks/flashinfer_benchmark.py" >&2
 printf 'Run setup first: %s/setup.sh\n' "${SCRIPT_DIR}" >&2
 exit 1
fi

cd "${FLASHINFER_DIR}"

for batch_size in "${BATCH_SIZE_LIST[@]}"; do
 output="$(
   python benchmarks/flashinfer_benchmark.py \
     --routine BatchMLAPagedAttentionWrapper \
     --backends "${BACKEND}" \
     --page_size "${PAGE_SIZE}" \
     --batch_size "${batch_size}" \
     --s_qo "${S_QO}" \
     --s_kv "${S_KV}" \
     --num_qo_heads "${NUM_QO_HEADS}" \
     --num_kv_heads "${NUM_KV_HEADS}" \
     --head_dim_ckv "${HEAD_DIM_CKV}" \
     --head_dim_kpe "${HEAD_DIM_KPE}" \
     --q_dtype "${Q_DTYPE}" \
     --kv_dtype "${KV_DTYPE}" \
     --num_iters "${NUM_ITERS}" \
     --dry_run_iters "${DRY_RUN_ITERS}" \
     -vv 2>&1
 )" || {
   printf 'batch_size=%s ERROR\n' "${batch_size}" >&2
   printf '%s\n' "${output}" >&2
   exit 1
 }
 last_line="$(printf '%s\n' "${output}" | awk 'NF { line = $0 } END { print line }')"
 printf 'batch_size=%s %s\n' "${batch_size}" "${last_line}"
done
