# Full-Chain Training Pipeline

This document describes the public training path for the Stage C TokenGraph-LLM line. The GitHub release asset contains the model package only. The source repository contains the algorithms, data conversion scripts, graph builders, training code, and evaluation templates.

## Pipeline

```text
open corpus
  -> schema2 JSONL records
  -> optional semantic teacher annotation
  -> token-level reasoning graph dataset
  -> Stage C Dynamic Token Graph Decoder V3 training
  -> graph ablation / attribution evaluation
  -> external model package release
```

## 1. Install Dependencies

Core model training:

```bash
pip install -r requirements.txt
```

Full-chain data processing and optional teacher annotation:

```bash
pip install -r requirements-full-chain.txt
```

`transformers` is only needed when you use a local Hugging Face causal LM as a semantic teacher. The TokenGraph-LLM architecture itself is not a Transformer wrapper.

## 2. Convert Open Corpora To Schema2

Schema2 is the public token-language corpus format used before graph construction. Required fields are `query`, `source_segments`, `text_units`, and `target_text`.

Long text parquet corpora:

```bash
python scripts/build_schema2_from_open_longtext_parquets.py \
  --parquet-root /data/open_longtext_parquets \
  --out-jsonl /data/schema2/open_longtext.schema2.jsonl \
  --dataset-label open_longtext \
  --max-records 1000000
```

General ability corpora from Hugging Face:

```bash
python scripts/build_general_ability_schema2_from_hf.py \
  --out-jsonl /data/schema2/general_ability.schema2.jsonl \
  --max-records 24000 \
  --streaming
```

Parquet-based general ability conversion:

```bash
python scripts/build_general_ability_schema2_from_hf_parquets.py \
  --out-jsonl /data/schema2/general_ability.parquet.schema2.jsonl \
  --cache-dir /data/hf_cache \
  --max-records 28000
```

Long continuation multitask expansion:

```bash
python scripts/compose_long_multitask_schema2.py \
  --input-jsonl /data/schema2/open_longtext.schema2.jsonl \
  --out-jsonl /data/schema2/open_longtext.multitask.schema2.jsonl \
  --max-output-records 30000
```

## 3. Optional Teacher Annotation

The optional teacher adds semantic spans and typed edges before graph building. It is training-time preprocessing only; inference does not call the teacher.

OpenAI-compatible endpoint:

```bash
export TOKEN_SEMANTIC_TEACHER_API_KEY=...

python scripts/annotate_token_semantic_graph_with_openai.py \
  --input-jsonl /data/schema2/open_longtext.schema2.jsonl \
  --out-jsonl /data/schema2/open_longtext.annotated.schema2.jsonl \
  --progress-json /data/schema2/teacher_progress.json \
  --base-url https://example.com/v1 \
  --model gpt-5-mini \
  --api-key-env TOKEN_SEMANTIC_TEACHER_API_KEY \
  --workers 4
```

Local Hugging Face teacher:

```bash
python scripts/annotate_token_semantic_graph_with_local_hf.py \
  --input-jsonl /data/schema2/open_longtext.schema2.jsonl \
  --out-jsonl /data/schema2/open_longtext.local_teacher.schema2.jsonl \
  --progress-json /data/schema2/local_teacher_progress.json \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --batch-size 4 \
  --dtype bf16
```

## 4. Build Token Graph Dataset

```bash
python scripts/build_native_token_reasoning_graph_dataset_v3.py \
  --input-jsonl /data/schema2/open_longtext.annotated.schema2.jsonl \
  --out-dir /data/token_graph_dataset_v3 \
  --tokenizer-kind hf_bpe \
  --pretrained-tokenizer gpt2 \
  --vocab-size 50257 \
  --tokenizer-text-limit 10000 \
  --tokenizer-char-budget 2000000 \
  --graph-mode simple_plus_causal_target \
  --max-target-prefix-tokens 160 \
  --progress-every 1000
```

Expected outputs:

```text
tokenizer.json
train.base.jsonl
val.base.jsonl
annotation_input.jsonl
manifest.json
```

For larger corpora, use:

```bash
python scripts/build_native_token_reasoning_graph_dataset_v3_parallel.py ...
python scripts/build_native_token_reasoning_graph_dataset_v3_resume_spill.py ...
```

## 5. Train Stage C

```bash
cd src/token_graph_llm
python train_token_graph_dynamic_decoder_v3.py \
  --dataset-dir /data/token_graph_dataset_v3 \
  --out-dir /runs/stagec_v3 \
  --streaming-train \
  --max-steps 62000 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --dim 512 \
  --graph-layers 8 \
  --decoder-layers 10 \
  --heads 8 \
  --untie-embeddings \
  --amp bf16 \
  --lr 0.0002 \
  --label-smoothing 0.02 \
  --graph-state-weight 0.35 \
  --next-token-node-weight 0.08 \
  --edge-type-weight 0.05
```

Main outputs:

```text
token_graph_dynamic_decoder_v3.pt
summary.json
```

## 6. End-To-End Templates

Single dataset template:

```bash
RAW_SCHEMA2_JSONL=/data/schema2/open_longtext.schema2.jsonl \
WORK_DIR=/runs/stagec_full_chain \
GRAPH_MODE=simple_plus_causal_target \
bash scripts/run_stagec_full_chain_template.sh
```

Sharded large-corpus template:

```bash
RAW_SCHEMA2_JSONL=/data/schema2/open_longtext.schema2.jsonl \
WORK_DIR=/runs/stagec_sharded \
SHARD_SIZE=50000 \
TOTAL_RECORDS=1000000 \
GRAPH_MODE=simple_plus_causal_target \
bash scripts/run_stagec_sharded_training_template.sh
```

## 7. Evaluate Graph Dependence

```bash
cd src/token_graph_llm
python eval_dynamic_v3_compare_ablation.py \
  --dataset-dir /data/token_graph_dataset_v3 \
  --checkpoints-json /runs/stagec_v3/checkpoints.json \
  --out-json /runs/stagec_v3/stagec_ablation_eval.json \
  --out-html /runs/stagec_v3/stagec_ablation_attribution.html
```

The evaluation compares normal, no-edge, and shuffled-edge graph variants and emits attribution HTML for generated tokens.

## Reproducibility Notes

- The source tree does not include raw corpora or `.pt` checkpoint files.
- API keys must be supplied by environment variable and are not stored in configs.
- Internal machine paths and private run logs are intentionally excluded.
- Release assets are model packages only; source updates live in Git.
