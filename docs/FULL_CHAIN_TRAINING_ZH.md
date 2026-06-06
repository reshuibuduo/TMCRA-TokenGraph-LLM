# 全链路训练流程

本文档说明 Stage C TokenGraph-LLM 的公开训练链路。GitHub Release 资产只放模型包；源码仓库放算法、数据转换、建图、训练、评测模板。

## 总流程

```text
开源语料
  -> schema2 JSONL 记录
  -> 可选 teacher 语义标注
  -> token-level reasoning graph dataset
  -> Stage C Dynamic Token Graph Decoder V3 训练
  -> 图消融 / attribution 评测
  -> 独立模型包发布
```

## 1. 安装依赖

核心模型训练：

```bash
pip install -r requirements.txt
```

全链路数据处理和可选 teacher 标注：

```bash
pip install -r requirements-full-chain.txt
```

`transformers` 只在使用本地 Hugging Face 模型做训练前 teacher 标注时需要。TokenGraph-LLM 本体不是 Transformer 外壳。

## 2. 语料转换为 Schema2

Schema2 是建图前的公开 token-language 语料格式。核心字段是 `query`、`source_segments`、`text_units`、`target_text`。

长文本 parquet 语料：

```bash
python scripts/build_schema2_from_open_longtext_parquets.py \
  --parquet-root /data/open_longtext_parquets \
  --out-jsonl /data/schema2/open_longtext.schema2.jsonl \
  --dataset-label open_longtext \
  --max-records 1000000
```

Hugging Face 通用能力语料：

```bash
python scripts/build_general_ability_schema2_from_hf.py \
  --out-jsonl /data/schema2/general_ability.schema2.jsonl \
  --max-records 24000 \
  --streaming
```

parquet 版通用能力转换：

```bash
python scripts/build_general_ability_schema2_from_hf_parquets.py \
  --out-jsonl /data/schema2/general_ability.parquet.schema2.jsonl \
  --cache-dir /data/hf_cache \
  --max-records 28000
```

长续写多任务扩展：

```bash
python scripts/compose_long_multitask_schema2.py \
  --input-jsonl /data/schema2/open_longtext.schema2.jsonl \
  --out-jsonl /data/schema2/open_longtext.multitask.schema2.jsonl \
  --max-output-records 30000
```

## 3. 可选 Teacher 标注

teacher 会在建图前补充 semantic spans 和 typed edges。它只用于训练前处理，推理阶段不会调用 teacher。

OpenAI 兼容接口：

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

本地 Hugging Face teacher：

```bash
python scripts/annotate_token_semantic_graph_with_local_hf.py \
  --input-jsonl /data/schema2/open_longtext.schema2.jsonl \
  --out-jsonl /data/schema2/open_longtext.local_teacher.schema2.jsonl \
  --progress-json /data/schema2/local_teacher_progress.json \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --batch-size 4 \
  --dtype bf16
```

## 4. 构建 Token Graph Dataset

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

预期输出：

```text
tokenizer.json
train.base.jsonl
val.base.jsonl
annotation_input.jsonl
manifest.json
```

大语料可使用：

```bash
python scripts/build_native_token_reasoning_graph_dataset_v3_parallel.py ...
python scripts/build_native_token_reasoning_graph_dataset_v3_resume_spill.py ...
```

## 5. 训练 Stage C

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

主要输出：

```text
token_graph_dynamic_decoder_v3.pt
summary.json
```

## 6. 端到端模板

单数据集：

```bash
RAW_SCHEMA2_JSONL=/data/schema2/open_longtext.schema2.jsonl \
WORK_DIR=/runs/stagec_full_chain \
GRAPH_MODE=simple_plus_causal_target \
bash scripts/run_stagec_full_chain_template.sh
```

大语料分片训练：

```bash
RAW_SCHEMA2_JSONL=/data/schema2/open_longtext.schema2.jsonl \
WORK_DIR=/runs/stagec_sharded \
SHARD_SIZE=50000 \
TOTAL_RECORDS=1000000 \
GRAPH_MODE=simple_plus_causal_target \
bash scripts/run_stagec_sharded_training_template.sh
```

## 7. 图依赖评测

```bash
cd src/token_graph_llm
python eval_dynamic_v3_compare_ablation.py \
  --dataset-dir /data/token_graph_dataset_v3 \
  --checkpoints-json /runs/stagec_v3/checkpoints.json \
  --out-json /runs/stagec_v3/stagec_ablation_eval.json \
  --out-html /runs/stagec_v3/stagec_ablation_attribution.html
```

评测会比较 normal、no_edges、shuffle_edges，并输出 generated token 的 attribution HTML。

## 复现说明

- 源码仓库不包含原始语料和 `.pt` checkpoint。
- API key 必须通过环境变量传入，不写入配置。
- 内部机器路径和私有运行日志不进入开源包。
- Release 资产只放模型包；源码更新走 Git。
