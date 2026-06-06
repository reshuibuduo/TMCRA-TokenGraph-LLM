# Stage C 详细测试记录 2026-06-06

对象：`train_dynamic_v3_stageC_full1m_stream_bf16_b4g4_w8_512_g8_d10_20260606`

Checkpoint：`runs/train_dynamic_v3_stageC_full1m_stream_bf16_b4g4_w8_512_g8_d10_20260606_060301/token_graph_dynamic_decoder_v3.pt`

说明：本记录是 smoke 级评估，不是正式论文分数。目的不是证明模型已经可用，而是判断 Stage C 是否已经出现可测的语言能力、图边依赖和语法偏好。

## 可跑的小模型 Benchmark

当前适合 TMCRA TokenGraph-LLM 早期阶段的 benchmark：

1. TinyStories validation smoke
   - 类型：故事续写生成。
   - 适合原因：面向小语言模型，能直接观察自然语言连贯性。
   - 本次用 `TinyStories-valid.txt`，通过 `HF_ENDPOINT=https://hf-mirror.com` 从镜像站拉取。

2. BLiMP likelihood smoke
   - 类型：最小对语法判断。
   - 适合原因：不要求聊天能力，只比较 good sentence / bad sentence 的平均 token loss。
   - 本次跑 3 个子类，每类 100 对。

3. BabyLM / BLiMP 全套
   - 类型：小模型语言能力综合评估。
   - 当前状态：可以做，但需要把 TMCRA TokenGraph-LLM 封装成完整 likelihood adapter，后续可接入更标准的 evaluation pipeline。

## TinyStories Validation Smoke

输出文件：`tinystories_stagec_smoke.json`

该文件可作为本地 eval output 或 release asset 中的附加评估记录保存。

设置：

- 样本数：8
- prompt 长度：45 words
- max_new_tokens：160
- temperature：0.9
- top_k：40
- 对照：normal / no_edges / shuffle_edges

| variant | avg_words | unique_word_ratio | repeated_bigrams | avg_gold_overlap | avg_normal_overlap |
|---|---:|---:|---:|---:|---:|
| normal | 73.88 | 0.8089 | 0.75 | 0.1835 | 1.0000 |
| no_edges | 38.12 | 0.8806 | 0.12 | 0.1499 | 0.1530 |
| shuffle_edges | 63.62 | 0.8092 | 0.62 | 0.1618 | 0.2073 |

结论：

- normal 图下生成显著更长，断边后平均长度从 73.88 words 降到 38.12 words。
- no_edges 和 shuffle_edges 与 normal 的文本重合度低，说明 Stage C 输出不是单纯由 decoder 语言先验决定，图边会改变生成轨迹。
- 但 avg_gold_overlap 仍低，续写会保留部分实体或场景，却经常改写事实走向。
- 当前表现更像“已经能沿图生成相似童话风格文本”，不是“能稳定跟随原故事事实续写”。

案例：sad cow

Prompt：

> Once upon a time, there was a kind farmer. He had a big cow. The cow was sad. The farmer did not know why. One day, a little boy came to the farm. He saw the sad cow. The boy kneeled down to talk to

Gold prefix：

> the cow. "Why are you sad, cow?" he asked. The cow said, "I am lonely. I want a friend." ...

Normal：

> his friend, the little boy. The little boy was happy to see him. He said, "Don't worry, I will help you." So, the farmer put his hands on the cow's tree. The little girl was happy and played with all day. They became best friends...

判断：

- 优点：保留了 cow / farmer / little boy / help / friends 这条语义链。
- 问题：局部事实不稳定，例如 `cow's tree`、人物性别和动作漂移。

## BLiMP Likelihood Smoke

输出目录：`stagec_blimp_smoke_20260606/`

该目录可作为本地 eval output 或 release asset 中的附加评估记录保存。

方法：

- good sentence 和 bad sentence 分别作为 `target_text`。
- 使用相同 instruction graph：`Generate a grammatical English sentence.`
- 计算平均 token cross entropy。
- good_loss < bad_loss 记为正确。

| BLiMP 子类 | 样本数 | 正确数 | accuracy |
|---|---:|---:|---:|
| determiner_noun_agreement_1 | 100 | 59 | 59% |
| anaphor_number_agreement | 100 | 63 | 63% |
| regular_plural_subject_verb_agreement_1 | 100 | 64 | 64% |

正确案例：

- good：`Craig explored that grocery store.` loss 6.814184
- bad：`Craig explored that grocery stores.` loss 7.182649
- 判断：正确，模型偏向单数 determiner-noun 一致。

- good：`Most legislatures haven't disliked children.` loss 7.407795
- bad：`Most legislatures hasn't disliked children.` loss 7.738126
- 判断：正确，模型偏向复数主语搭配 `haven't`。

错误案例：

- good：`Raymond is selling this sketch.` loss 6.946195
- bad：`Raymond is selling this sketches.` loss 6.843910
- 判断：错误，模型没有稳定绑定 `this + singular noun`。

- good：`Paula references Robert.` loss 7.911676
- bad：`Paula reference Robert.` loss 7.645935
- 判断：错误，第三人称单数动词偏好不稳定。

## 当前 Stage C 能力边界

已经出现的能力：

- 能生成比 Stage A/B 更长、更连贯的英文。
- 图边对生成有明显影响；断边/乱边会改变长度和内容。
- 在 BLiMP 部分最小对任务上高于随机，说明有弱语法偏好。

仍然暴露的问题：

- 事实绑定弱：prompt 里有正确实体，但生成时容易混入同风格实体或动作。
- 语法能力不稳定：BLiMP 只到 59%-64%，距离可靠语言模型还远。
- 长生成会漂移：故事风格保留，但事件链经常跑偏。
- 当前图边更像“生成轨迹调制器”，还没有稳定成为“事实约束器”。

## 下一步建议

1. 补 TMCRA TokenGraph-LLM 的正式 likelihood adapter
   - 支持批量 BLiMP 全套、BabyLM 子集、WikiText/TinyStories perplexity。

2. 加强 graph-to-token grounding loss
   - 当前 normal 优于 no_edges，说明图有效；但图没有稳定约束事实。
   - 需要让每个生成位置更稳定绑定对应 source token / relation path。

3. 对 Stage C 做生成归因批量统计
   - 统计生成 token 的 top graph node 是否来自 prompt 中的关键实体。
   - 重点看错误续写时是否跳到了错误节点。

4. 后续训练不要只扩大语料
   - 现在问题不只是语料量，而是图边到 token 输出的约束还不够强。
   - 需要在训练目标里增加 fact path consistency / graph-token alignment。
