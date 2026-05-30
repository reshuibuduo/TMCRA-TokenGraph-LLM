# TMCRA TokenGraph-LLM

[![GitHub Repo](https://img.shields.io/badge/GitHub-source-181717?logo=github)](https://github.com/reshuibuduo/TMCRA-TokenGraph-LLM)
[![Model Release](https://img.shields.io/badge/GitHub-Release-blue?logo=github)](https://github.com/reshuibuduo/TMCRA-TokenGraph-LLM/releases/tag/v0.1.0-prototype)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model-yellow?logo=huggingface)](https://huggingface.co/2009YU/TMCRA-TokenGraph-LLM)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

TMCRA TokenGraph-LLM 是一个实验性的图原生自回归语言模型原型。它不是 Transformer 外壳，也不会在推理时调用外部 LLM。文本生成来自 token 级图编码、图消息传递和图结构因果解码器。

这个仓库是用于开源审阅的研究原型包。它展示了 token 级图模型可以通过 next-token prediction 和图路径目标进行训练，并且可以对每个生成 token 做 graph node / edge attribution。它不是成熟 SDK，也不是生产可用 LLM。

## 项目作用

- 将文本或指令语料构造成 token-level graph。
- 训练不依赖 Transformer self-attention 的图原生自回归解码器。
- 以 next-token prediction 作为主训练目标。
- 增加可选图结构目标：
  - token path alignment
  - token transition path consistency
  - relation transition loss
  - causal path consistency loss
  - 非 EOS 正则，用来缓解过早停止和短答塌缩
- 提供 token attribution：生成 token -> top graph nodes -> incident graph edges。

## 当前状态

当前原型可以学习英文 token 分布，并生成短自然语言片段，但还不是可用的通用 LLM。主要短板包括指令跟随、精确事实回答、长程一致性、多语言生成和概念绑定。

目前相对较好的能力是短故事续写。较弱的是精确问答、数字/事实回答、结构化列表和抽象定义。

## 已发布模型

当前 checkpoint 与源码分开发布：

[![下载模型包](https://img.shields.io/badge/Download-model_package.zip-2ea44f)](https://github.com/reshuibuduo/TMCRA-TokenGraph-LLM/releases/download/v0.1.0-prototype/token_graph_llm_model_package_20260530.zip)
[![Hugging Face 模型页](https://img.shields.io/badge/View_on-Hugging_Face-yellow?logo=huggingface)](https://huggingface.co/2009YU/TMCRA-TokenGraph-LLM)

源码仓库默认不包含 `.pt` checkpoint 和原始训练语料。Release 模型包包含 checkpoint、tokenizer、model card、manifest、checksum 和样例输出。

## 当前训练规模和样例

已发布 checkpoint 基于 token-graph 数据集训练：

- 训练样本：`920,048`
- 验证样本：`80,004`
- 词表大小：`1,012`
- 模型结构：`dim=384`，`graph_layers=6`，`decoder_layers=8`，untied output embedding
- 微调：额外 `3,000` steps，加入 relation-transition 和 causal-path 目标

当前能力仍处在早期阶段。模型可以生成短英文片段，尤其是故事续写类文本，但还不是可靠的事实问答模型。

贪心解码样例：

```text
Prompt:
Continue the story in natural language.

Output:
to her mom. "Mom, can I have some oats?" she asked. Her mom said,
"Yes, but be careful. That is very yummy!" Mia was so excited to eat
the oats that she wanted to show her friends.
```

弱项样例：

```text
Prompt:
How deep was the water that rushed through the school?

Output:
feetings.com food.
```

这个能力边界是刻意写清楚的：本包主要展示图原生语言模型架构、训练路径和归因工具，不宣称已经达到成熟 LLM 水平。

## 目录结构

```text
src/token_graph_llm/
  native_token_graph_common.py       tokenizer 工具
  token_graph_llm_model_v1.py        图编码器、图因果解码器、训练 loss
  train_token_graph_llm_v1.py        训练和微调入口
  generalization_eval_probe_v1.py    泛化/反复制测试
  token_attribution_v1.py            token 级图归因可视化

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
  "source_segments": ["optional supporting text"],
  "text_units": ["optional unit-level text spans"],
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

输出：

```text
tokenizer.json
train.base.jsonl
val.base.jsonl
annotation_input.jsonl
manifest.json
```

大规模语料可以使用 `scripts/` 下的 parallel/resume-spill builder。

## 训练

在 `src/token_graph_llm` 目录运行：

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

训练输出：

```text
token_graph_llm_v1.pt
summary.json
```

## 从 checkpoint 继续训练

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

继续训练时，模型结构参数必须和 checkpoint 匹配。

## 泛化测试

```bash
python generalization_eval_probe_v1.py \
  --run-dir /path/to/run_out \
  --dataset-dir /path/to/dataset_out \
  --prompts-jsonl ../../examples/generalization_probe_prompts.jsonl \
  --out-json /path/to/run_out/generalization_probe_v1.json \
  --train-neighbor-scan 10000
```

该测试会输出 nearest train similarity、copy ratio、new-token ratio 和 repetition ratio，用于判断模型是否只是在复制训练样本。

## Token Attribution

```bash
python token_attribution_v1.py \
  --run-dir /path/to/run_out \
  --dataset-dir /path/to/dataset_out \
  --out-json /path/to/run_out/token_attribution_v1.json \
  --out-html /path/to/run_out/token_attribution_v1.html
```

HTML 会展示每个生成 token 对应的 top graph nodes 和 incident edges。

## 模型文件

源码包默认不包含 checkpoint。请使用上方 Release 链接中的模型包，或将兼容 checkpoint 作为单独 GitHub Release / Hugging Face model file 发布。

## 安全和隐私

本包只应包含源码、文档和小型示例。正式发布前需要做本地 secret scan，确认没有密钥、内部主机名、私有日志或原始训练数据 dump。

## 许可证

MIT License。见 `LICENSE`。
