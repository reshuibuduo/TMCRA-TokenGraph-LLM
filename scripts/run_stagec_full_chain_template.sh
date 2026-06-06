#!/usr/bin/env bash
set -euo pipefail

# Public, parameterized Stage C full-chain template.
# It intentionally contains no private hostnames, absolute run paths, API keys,
# or raw corpus assumptions.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"

: "${RAW_SCHEMA2_JSONL:?Set RAW_SCHEMA2_JSONL to a schema2 JSONL corpus.}"

WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/runs/stagec_full_chain}"
DATASET_DIR="${DATASET_DIR:-${WORK_DIR}/dataset_v3}"
RUN_DIR="${RUN_DIR:-${WORK_DIR}/train_stagec_v3}"
ANNOTATED_SCHEMA2_JSONL="${ANNOTATED_SCHEMA2_JSONL:-${WORK_DIR}/schema2.annotated.jsonl}"

TOKENIZER_KIND="${TOKENIZER_KIND:-hf_bpe}"
PRETRAINED_TOKENIZER="${PRETRAINED_TOKENIZER:-gpt2}"
VOCAB_SIZE="${VOCAB_SIZE:-50257}"
LIMIT="${LIMIT:-0}"
GRAPH_MODE="${GRAPH_MODE:-simple_plus_causal_target}"
MAX_TARGET_PREFIX_TOKENS="${MAX_TARGET_PREFIX_TOKENS:-160}"
MAX_STEPS="${MAX_STEPS:-62000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
DIM="${DIM:-512}"
GRAPH_LAYERS="${GRAPH_LAYERS:-8}"
DECODER_LAYERS="${DECODER_LAYERS:-10}"
HEADS="${HEADS:-8}"
AMP="${AMP:-bf16}"

ENABLE_TEACHER="${ENABLE_TEACHER:-0}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt-5-mini}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-TOKEN_SEMANTIC_TEACHER_API_KEY}"
TEACHER_WORKERS="${TEACHER_WORKERS:-4}"

mkdir -p "${WORK_DIR}" "${DATASET_DIR}" "${RUN_DIR}"

SCHEMA2_FOR_BUILD="${RAW_SCHEMA2_JSONL}"
if [[ "${ENABLE_TEACHER}" == "1" ]]; then
  if [[ -z "${TEACHER_BASE_URL}" ]]; then
    echo "ENABLE_TEACHER=1 requires TEACHER_BASE_URL." >&2
    exit 2
  fi
  "${PYTHON}" "${PROJECT_ROOT}/scripts/annotate_token_semantic_graph_with_openai.py" \
    --input-jsonl "${RAW_SCHEMA2_JSONL}" \
    --out-jsonl "${ANNOTATED_SCHEMA2_JSONL}" \
    --progress-json "${WORK_DIR}/teacher_progress.json" \
    --base-url "${TEACHER_BASE_URL}" \
    --model "${TEACHER_MODEL}" \
    --api-key-env "${TEACHER_API_KEY_ENV}" \
    --workers "${TEACHER_WORKERS}" \
    --limit "${LIMIT}"
  SCHEMA2_FOR_BUILD="${ANNOTATED_SCHEMA2_JSONL}"
fi

"${PYTHON}" "${PROJECT_ROOT}/scripts/build_native_token_reasoning_graph_dataset_v3.py" \
  --input-jsonl "${SCHEMA2_FOR_BUILD}" \
  --out-dir "${DATASET_DIR}" \
  --limit "${LIMIT}" \
  --tokenizer-kind "${TOKENIZER_KIND}" \
  --pretrained-tokenizer "${PRETRAINED_TOKENIZER}" \
  --vocab-size "${VOCAB_SIZE}" \
  --tokenizer-text-limit 10000 \
  --tokenizer-char-budget 2000000 \
  --graph-mode "${GRAPH_MODE}" \
  --max-target-prefix-tokens "${MAX_TARGET_PREFIX_TOKENS}" \
  --progress-every 1000

(
  cd "${PROJECT_ROOT}/src/token_graph_llm"
  "${PYTHON}" train_token_graph_dynamic_decoder_v3.py \
    --dataset-dir "${DATASET_DIR}" \
    --out-dir "${RUN_DIR}" \
    --streaming-train \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --dim "${DIM}" \
    --graph-layers "${GRAPH_LAYERS}" \
    --decoder-layers "${DECODER_LAYERS}" \
    --heads "${HEADS}" \
    --untie-embeddings \
    --amp "${AMP}" \
    --lr 0.0002 \
    --label-smoothing 0.02 \
    --graph-state-weight 0.35 \
    --next-token-node-weight 0.08 \
    --edge-type-weight 0.05
)

cat > "${RUN_DIR}/checkpoints.json" <<JSON
{
  "StageC": "${RUN_DIR}/token_graph_dynamic_decoder_v3.pt"
}
JSON

(
  cd "${PROJECT_ROOT}/src/token_graph_llm"
  "${PYTHON}" eval_dynamic_v3_compare_ablation.py \
    --dataset-dir "${DATASET_DIR}" \
    --checkpoints-json "${RUN_DIR}/checkpoints.json" \
    --out-json "${RUN_DIR}/stagec_ablation_eval.json" \
    --out-html "${RUN_DIR}/stagec_ablation_attribution.html" \
    --sample-limit 8 \
    --metric-limit 64 \
    --max-val-batches 8
)

echo "Stage C full-chain run completed."
echo "Dataset: ${DATASET_DIR}"
echo "Run: ${RUN_DIR}"
