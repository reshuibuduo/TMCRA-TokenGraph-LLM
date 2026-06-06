#!/usr/bin/env bash
set -euo pipefail

# Public shard-by-shard training template for large corpora.
# Each shard is converted to a token graph dataset and trained before moving to
# the next shard. This avoids keeping a huge built graph dataset on disk.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"

: "${RAW_SCHEMA2_JSONL:?Set RAW_SCHEMA2_JSONL to a schema2 JSONL corpus.}"

WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/runs/stagec_sharded}"
RUN_ROOT="${RUN_ROOT:-${WORK_DIR}/runs}"
SHARD_ROOT="${SHARD_ROOT:-${WORK_DIR}/shards}"
SHARD_SIZE="${SHARD_SIZE:-50000}"
TOTAL_RECORDS="${TOTAL_RECORDS:-1000000}"
MAX_STEPS_PER_SHARD="${MAX_STEPS_PER_SHARD:-3000}"

DIM="${DIM:-512}"
GRAPH_LAYERS="${GRAPH_LAYERS:-8}"
DECODER_LAYERS="${DECODER_LAYERS:-10}"
HEADS="${HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
AMP="${AMP:-bf16}"
TOKENIZER_KIND="${TOKENIZER_KIND:-hf_bpe}"
PRETRAINED_TOKENIZER="${PRETRAINED_TOKENIZER:-gpt2}"
VOCAB_SIZE="${VOCAB_SIZE:-50257}"
GRAPH_MODE="${GRAPH_MODE:-simple_plus_causal_target}"
MAX_TARGET_PREFIX_TOKENS="${MAX_TARGET_PREFIX_TOKENS:-160}"

mkdir -p "${WORK_DIR}" "${RUN_ROOT}" "${SHARD_ROOT}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
offset=0
shard_index=0
while [[ "${offset}" -lt "${TOTAL_RECORDS}" ]]; do
  shard_id="$(printf "%05d" "${shard_index}")"
  shard_dataset="${SHARD_ROOT}/dataset_${shard_id}"
  shard_run="${RUN_ROOT}/run_${shard_id}"
  shard_limit="${SHARD_SIZE}"

  echo "Building shard ${shard_id}: offset=${offset}, limit=${shard_limit}"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/build_native_token_reasoning_graph_dataset_v3.py" \
    --input-jsonl "${RAW_SCHEMA2_JSONL}" \
    --out-dir "${shard_dataset}" \
    --limit "${shard_limit}" \
    --skip-records "${offset}" \
    --tokenizer-kind "${TOKENIZER_KIND}" \
    --pretrained-tokenizer "${PRETRAINED_TOKENIZER}" \
    --vocab-size "${VOCAB_SIZE}" \
    --tokenizer-text-limit 10000 \
    --tokenizer-char-budget 2000000 \
    --graph-mode "${GRAPH_MODE}" \
    --max-target-prefix-tokens "${MAX_TARGET_PREFIX_TOKENS}" \
    --progress-every 1000

  train_args=(
    train_token_graph_dynamic_decoder_v3.py
    --dataset-dir "${shard_dataset}"
    --out-dir "${shard_run}"
    --streaming-train
    --max-steps "${MAX_STEPS_PER_SHARD}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --dim "${DIM}"
    --graph-layers "${GRAPH_LAYERS}"
    --decoder-layers "${DECODER_LAYERS}"
    --heads "${HEADS}"
    --untie-embeddings
    --amp "${AMP}"
    --lr 0.0002
    --label-smoothing 0.02
    --graph-state-weight 0.35
    --next-token-node-weight 0.08
    --edge-type-weight 0.05
  )
  if [[ -n "${INIT_CHECKPOINT}" ]]; then
    train_args+=(--init-checkpoint "${INIT_CHECKPOINT}")
  fi

  (
    cd "${PROJECT_ROOT}/src/token_graph_llm"
    "${PYTHON}" "${train_args[@]}"
  )

  INIT_CHECKPOINT="${shard_run}/token_graph_dynamic_decoder_v3.pt"
  echo "${INIT_CHECKPOINT}" > "${WORK_DIR}/latest_checkpoint.txt"
  offset=$((offset + shard_limit))
  shard_index=$((shard_index + 1))
done

echo "Sharded Stage C training completed."
echo "Latest checkpoint: ${INIT_CHECKPOINT}"
