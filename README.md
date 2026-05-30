# TMCRA TokenGraph-LLM

TMCRA TokenGraph-LLM is an experimental graph-native autoregressive language model prototype. It is not a Transformer wrapper and does not call an external LLM at inference time. Text is generated from token-level graph encoding, graph message passing, and a graph causal decoder.

This repository is a research prototype package prepared for open-source review. It demonstrates that a token-level graph model can be trained with next-token prediction plus graph path objectives, and that each generated token can be inspected through graph-node attribution. It is not a polished SDK and it is not a production LLM.

## What This Project Does

- Builds token-level graphs from text and instruction-style corpora.
- Trains a graph-native autoregressive decoder without Transformer self-attention.
- Keeps next-token prediction as the main objective.
- Adds optional graph objectives:
  - token path alignment
  - token transition path consistency
  - relation transition loss
  - causal path consistency loss
  - non-EOS regularization for early-stop collapse
- Provides token attribution for generated text: generated token -> top graph nodes -> incident graph edges.

## Current Status

The current prototype can learn English-like token distributions and produce short natural-language fragments. It is not yet a usable general LLM. Current weaknesses include instruction following, exact factual answering, long-range coherence, multilingual generation, and robust concept binding.

The strongest current behavior is short story-style continuation. The weakest behavior is precise QA, numeric/factual answers, structured lists, and abstract definitions.

## Released Checkpoint

The current prototype checkpoint is published separately from the source tree:

- GitHub Release: [v0.1.0 prototype checkpoint package](https://github.com/reshuibuduo/TMCRA-TokenGraph-LLM/releases/tag/v0.1.0-prototype)
- Model asset: [`token_graph_llm_model_package_20260530.zip`](https://github.com/reshuibuduo/TMCRA-TokenGraph-LLM/releases/download/v0.1.0-prototype/token_graph_llm_model_package_20260530.zip)
- Hugging Face model repo: [2009YU/TMCRA-TokenGraph-LLM](https://huggingface.co/2009YU/TMCRA-TokenGraph-LLM)

The source repository intentionally excludes `.pt` checkpoints and raw corpora. The release package includes the checkpoint, tokenizer, model card, manifest, checksums, and example outputs.

## Current Training Scale And Examples

The released checkpoint continues from a base model trained on a token-graph dataset with:

- train samples: `920,048`
- validation samples: `80,004`
- vocabulary size: `1,012`
- model shape: `dim=384`, `graph_layers=6`, `decoder_layers=8`, untied output embedding
- finetuning: `3,000` additional steps with relation-transition and causal-path objectives

Observed capability is still early-stage. It can generate short English fragments, especially story-like continuations, but it is not a reliable factual QA model.

Example greedy output:

```text
Prompt:
Continue the story in natural language.

Output:
to her mom. "Mom, can I have some oats?" she asked. Her mom said,
"Yes, but be careful. That is very yummy!" Mia was so excited to eat
the oats that she wanted to show her friends.
```

Weak-case example:

```text
Prompt:
How deep was the water that rushed through the school?

Output:
feetings.com food.
```

This boundary is intentional in the release notes: the package is meant to expose the graph-native language-model architecture, training path, and attribution tools, not to claim mature LLM performance.

## Repository Layout

```text
src/token_graph_llm/
  native_token_graph_common.py       tokenizer utilities
  token_graph_llm_model_v1.py        graph encoder + graph causal decoder + losses
  train_token_graph_llm_v1.py        training / finetuning entry point
  generalization_eval_probe_v1.py    anti-copy / generalization probe
  token_attribution_v1.py            token-level graph attribution

scripts/
  build_native_token_reasoning_graph_dataset_v3.py
  build_native_token_reasoning_graph_dataset_v3_parallel.py
  build_native_token_reasoning_graph_dataset_v3_resume_spill.py
  download_hf_sources.py

examples/
  generalization_probe_prompts.jsonl

docs/
  ARCHITECTURE_RUNTIME_ZH.md
  OPEN_CORPUS_10M_CANDIDATES.md

models/
  README.md
```

## Requirements

Python 3.10+ is recommended.

Install the minimal runtime:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build matching your CUDA environment from the official PyTorch instructions.

## Data Schema

The graph builder expects JSONL records with these preferred fields:

```json
{
  "query": "instruction or prompt",
  "source_segments": ["optional supporting text"],
  "text_units": ["optional unit-level text spans"],
  "target_text": "text to train the decoder to generate"
}
```

Legacy fields such as `answer`, `memory_nodes`, or `event_units` are compatibility paths only. New datasets should use `source_segments`, `text_units`, and `target_text`.

## Build A Small Dataset

Run from the `scripts` directory:

```bash
cd scripts
python build_native_token_reasoning_graph_dataset_v3.py \
  --input-jsonl /path/to/input.jsonl \
  --out-dir /path/to/dataset_out \
  --limit 3000 \
  --vocab-size 1024 \
  --min-pair-freq 5 \
  --tokenizer-kind hf_bpe \
  --tokenizer-text-limit 1000 \
  --tokenizer-char-budget 250000
```

Expected output:

```text
tokenizer.json
train.base.jsonl
val.base.jsonl
annotation_input.jsonl
manifest.json
```

For larger corpora, use the parallel/resume-spill builders in `scripts/`.

## Train

Run from `src/token_graph_llm`:

```bash
cd src/token_graph_llm
python train_token_graph_llm_v1.py \
  --dataset-dir /path/to/dataset_out \
  --out-dir /path/to/run_out \
  --streaming-train \
  --max-steps 3000 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --dim 384 \
  --graph-layers 6 \
  --decoder-layers 8 \
  --untie-embeddings \
  --lr 0.000025 \
  --label-smoothing 0.02 \
  --token-path-weight 0.02 \
  --transition-path-weight 0.02 \
  --relation-transition-weight 0.015 \
  --causal-path-weight 0.003 \
  --non-eos-weight 0.02 \
  --non-eos-steps 8
```

The run directory will contain:

```text
token_graph_llm_v1.pt
summary.json
```

## Continue From A Checkpoint

```bash
python train_token_graph_llm_v1.py \
  --dataset-dir /path/to/dataset_out \
  --out-dir /path/to/finetune_out \
  --init-checkpoint /path/to/token_graph_llm_v1.pt \
  --streaming-train \
  --max-steps 1000 \
  --dim 384 \
  --graph-layers 6 \
  --decoder-layers 8 \
  --untie-embeddings
```

The architecture parameters must match the checkpoint.

## Generalization Probe

```bash
python generalization_eval_probe_v1.py \
  --run-dir /path/to/run_out \
  --dataset-dir /path/to/dataset_out \
  --prompts-jsonl ../../examples/generalization_probe_prompts.jsonl \
  --out-json /path/to/run_out/generalization_probe_v1.json \
  --train-neighbor-scan 10000
```

This probe checks whether outputs are simple training-sample copies by reporting nearest training prompt similarity, prompt copy ratio, new-token ratio, and repetition ratio.

## Token Attribution

```bash
python token_attribution_v1.py \
  --run-dir /path/to/run_out \
  --dataset-dir /path/to/dataset_out \
  --out-json /path/to/run_out/token_attribution_v1.json \
  --out-html /path/to/run_out/token_attribution_v1.html
```

The HTML output shows top graph nodes and incident edges for each generated token.

## Model Checkpoints

Checkpoints are intentionally excluded from this source package. Use the released checkpoint package linked above, or publish compatible checkpoints as separate GitHub Release assets / Hugging Face model files.

## Security And Privacy

This package is intended to contain source code, documentation, and small examples only. Before publishing, run a local secret scan and confirm that no credentials, internal hostnames, private logs, or raw training dumps are included.

## License

MIT License. See `LICENSE`.
