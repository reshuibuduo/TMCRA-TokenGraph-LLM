# TMCRA TokenGraph-LLM

[![GitHub Repo](https://img.shields.io/badge/GitHub-source-181717?logo=github)](https://github.com/reshuibuduo/TMCRA-TokenGraph-LLM)
[![Stage C Release](https://img.shields.io/badge/GitHub-Stage_C_Release-blue?logo=github)](https://github.com/reshuibuduo/TMCRA-TokenGraph-LLM/releases/tag/v0.2.0-stagec)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model-yellow?logo=huggingface)](https://huggingface.co/2009YU/TMCRA-TokenGraph-LLM)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

TMCRA TokenGraph-LLM 是一个实验性的图原生自回归语言模型。它不是 Transformer 外壳，也不会在推理时调用外部 LLM。文本生成来自 token 级图编码、学习式边门控、图消息传递和动态图因果解码器。

当前默认路线是 **Stage C / Dynamic Token Graph Decoder V3**。Stage C 约 `114.6M` 参数，在百万级 token graph 语料上训练，联合使用 next-token、graph-state、tunnel、edge-type 和 next-token-node 目标。它仍然是研究原型，不是成熟 SDK，也不是生产可用 LLM。

## 项目作用

- 把文本和指令式语料构造成 token-level graph。
- 训练不依赖 Transformer self-attention 的图原生自回归 decoder。
- 让生成 token 本身也成为动态图节点。
- 学习 typed candidate edges 的边激活，而不是把图边当固定规则。
- 保持 next-token prediction 为主目标。
- 增加图结构训练目标：
  - graph-state token prediction
  - support-node scoring
  - answer-overlap scoring
  - decoder-to-context tunnel alignment
  - next-token-to-node alignment
  - edge-type prediction
- 提供 token attribution：generated token -> top graph nodes -> incident graph edges。

## 当前状态

Stage C 比旧 v0.1 checkpoint 能生成更长的英文文本。图消融显示 typed graph edges 会实质影响生成结果，不是装饰性结构。

当前仍不是可用通用 LLM。主要短板包括：

- 精确事实问答；
- 稳定长程一致性；
- 强指令跟随；
- 强语法能力；
- 多语言生成；
- 可靠概念绑定。

目前相对最好的行为是故事类续写；较弱的是精确 QA、数值/事实回答、结构化列表和抽象定义。

## 已发布模型

当前 Stage C checkpoint 与源码分开发布：

[![下载 Stage C 模型包](https://img.shields.io/badge/Download-stagec_model_package.zip-2ea44f)](https://github.com/reshuibuduo/TMCRA-TokenGraph-LLM/releases/download/v0.2.0-stagec/tgclm_stagec_model_package_20260606.zip)
[![Hugging Face 模型页](https://img.shields.io/badge/View_on-Hugging_Face-yellow?logo=huggingface)](https://huggingface.co/2009YU/TMCRA-TokenGraph-LLM)

源码仓库默认不包含 `.pt` checkpoint 和原始训练语料。Release 模型包包含 checkpoint、tokenizer、dataset manifest、training summary、checksum 和评估说明。

旧版本：

- `v0.1.0-prototype` 现在标记为 legacy small prototype checkpoint package。

## Stage C 训练规模

Stage C checkpoint 训练配置：

- 参数量：`114,615,372`
- 模型结构：`dim=512`，`graph_layers=8`，`decoder_layers=10`
- embedding：untied
- 精度：`bf16`
- 有效训练样本：约 `1.03M`
- 训练步数：`62,000`
- checkpoint：`token_graph_dynamic_decoder_v3.pt`
- tokenizer：随 Stage C dataset manifest 一起打包

## 当前 Smoke 测试

Stage A/B/C loss 与图消融：

| model | variant | total loss | lm loss |
|---|---|---:|---:|
| StageA | normal | 10.666509 | 7.587883 |
| StageB | normal | 10.228030 | 7.297534 |
| StageC | normal | 6.512117 | 4.641285 |
| StageC | no_edges | 8.310654 | 5.790666 |
| StageC | shuffle_edges | 7.702783 | 5.169387 |

TinyStories validation smoke：

| variant | avg words | avg gold overlap |
|---|---:|---:|
| normal | 73.88 | 0.1835 |
| no_edges | 38.12 | 0.1499 |
| shuffle_edges | 63.62 | 0.1618 |

BLiMP likelihood smoke：

| task | accuracy |
|---|---:|
| determiner_noun_agreement_1 | 59% |
| anaphor_number_agreement | 63% |
| regular_plural_subject_verb_agreement_1 | 64% |

这些只是 smoke 测试，不是榜单分数。它们说明 Stage C 已经有早期语言行为和图边依赖，同时也说明它还不是成熟 LLM。

## 目录结构

```text
src/token_graph_llm/
  native_token_graph_common.py
  token_graph_llm_model_v1.py              legacy v0.1 model
  train_token_graph_llm_v1.py              legacy v0.1 trainer
  model_token_graph_dynamic_decoder_v3.py  Stage C model
  train_token_graph_dynamic_decoder_v3.py  Stage C trainer
  train_graph_causal_decoder_v2.py         dataset / collate helpers
  eval_dynamic_v3_compare_ablation.py
  eval_stagec_tinystories_smoke_v3.py
  eval_stagec_blimp_likelihood_v3.py
  generalization_eval_probe_v1.py
  token_attribution_v1.py

scripts/
  build_native_token_reasoning_graph_dataset_v3.py
  build_native_token_reasoning_graph_dataset_v3_parallel.py
  build_native_token_reasoning_graph_dataset_v3_resume_spill.py
  download_hf_sources.py

docs/
  TGCLM_STAGEC_TECHNICAL_OVERVIEW.md
  TGCLM_STAGEC_TECHNICAL_OVERVIEW_ZH.md
  STAGEC_DETAILED_BENCHMARK_SMOKE_20260606.md
  TOKEN_LEVEL_SEMANTIC_GRAPH_SCHEMA.md
  ARCHITECTURE_RUNTIME_ZH.md
  OPEN_CORPUS_10M_CANDIDATES.md

models/
  README.md
```

## 依赖环境

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

GPU 训练需要安装与你的 CUDA 环境匹配的 PyTorch。

## 数据格式

推荐 JSONL 字段：

```json
{
  "query": "instruction or prompt",
  "source_segments": [
    {"segment_id": "seg1", "text": "optional supporting text"}
  ],
  "text_units": [
    {"unit_id": "u1", "text": "optional unit-level text spans"}
  ],
  "target_text": "text to train the decoder to generate"
}
```

旧字段如 `answer`、`memory_nodes`、`event_units` 只作为兼容路径。新语料应优先使用 `source_segments`、`text_units`、`target_text`。

## 构建小型数据集

在 `scripts` 目录运行：

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

预期输出：

```text
tokenizer.json
train.base.jsonl
val.base.jsonl
annotation_input.jsonl
manifest.json
```

大规模语料可使用 `scripts/` 下的 parallel / resume-spill builder。

## Stage C 风格训练

在 `src/token_graph_llm` 目录运行：

```bash
cd src/token_graph_llm
python train_token_graph_dynamic_decoder_v3.py \
  --dataset-dir /path/to/dataset_out \
  --out-dir /path/to/run_out \
  --streaming-train \
  --max-steps 62000 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --dim 512 \
  --graph-layers 8 \
  --decoder-layers 10 \
  --untie-embeddings \
  --amp bf16 \
  --lr 0.0002 \
  --label-smoothing 0.02 \
  --graph-state-weight 0.35 \
  --next-token-node-weight 0.08 \
  --edge-type-weight 0.05
```

训练输出：

```text
token_graph_dynamic_decoder_v3.pt
summary.json
```

## 从 checkpoint 继续训练

```bash
python train_token_graph_dynamic_decoder_v3.py \
  --dataset-dir /path/to/dataset_out \
  --out-dir /path/to/finetune_out \
  --init-checkpoint /path/to/token_graph_dynamic_decoder_v3.pt \
  --streaming-train \
  --max-steps 1000 \
  --dim 512 \
  --graph-layers 8 \
  --decoder-layers 10 \
  --untie-embeddings
```

继续训练时，模型结构参数必须和 checkpoint 匹配。

## Token Attribution

Stage C 后续建议使用 v3 评估脚本：

```bash
python eval_dynamic_v3_compare_ablation.py \
  --dataset-dir /path/to/dataset_out \
  --checkpoints-json /path/to/checkpoints.json \
  --out-json /path/to/eval_results.json \
  --out-html /path/to/attribution.html
```

HTML 会展示每个 generated token 对应的 top graph nodes 和候选 next tokens。

## 模型文件

源码包默认不包含 checkpoint。请使用上方 Release 链接中的模型包，或将兼容 checkpoint 作为单独 GitHub Release / Hugging Face model file 发布。

## 安全和隐私

本包只应包含源码、文档和小型示例。正式发布前需要确认没有密钥、内部主机名、私有日志或原始训练数据 dump。

## 许可证

MIT License。见 `LICENSE`。
