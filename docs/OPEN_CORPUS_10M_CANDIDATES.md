# Token Graph LLM v1 - 10M Open Corpus Candidate Plan

Date: 2026-05-29

Goal: prepare candidate open datasets for scaling the graph-native language model from the current 1M-token-graph corpus toward a 10M-record corpus. This is for natural-language generation, not TMCRA memory recall.

## Selection Rules

- Prefer permissive or clearly documented open data licenses.
- Exclude LongMemEval and TMCRA benchmark outputs from training.
- Prefer text that trains general language ability: continuation, explanation, summarization, instruction following, QA, math reasoning, and long-form knowledge.
- Avoid making the corpus mostly synthetic chat; the graph LLM needs broad language modeling distribution.
- Convert every source into the same schema: `source_segments`, `text_units`, `target_text`, `target_tokens`, then graph-build into token nodes/edges.

## Recommended 10M Mix

| Tier | Dataset | Target records | Role | License/status note |
|---|---:|---:|---|---|
| A | HuggingFaceFW/fineweb-edu | 2,500,000 | educational web continuation / knowledge text | ODC-By |
| A | HuggingFaceTB/cosmopedia | 2,500,000 | synthetic textbooks, stories, blogposts, WikiHow-style content | Apache-2.0 |
| A | allenai/dolma | 1,500,000 | broad web/books/wiki/code/science text | ODC-By |
| B | Open-Orca/OpenOrca | 1,000,000 | reasoning/instruction traces | MIT |
| B | teknium/OpenHermes-2.5 | 800,000 | general instruction and chat | verify final redistribution terms before packaging |
| B | existing current sources | 1,200,000 | TinyStories, WikiText, CNN/DM, SQuAD, GSM8K, Dolly, CoQA/NarrativeQA | mixed; keep source manifest |
| Reserve | fineweb general or SlimPajama | 500,000 | fallback if any source is slow/unavailable | verify source license and quality filters |

Total target: 10,000,000 records.

## Why This Mix

FineWeb-Edu and Dolma provide broad pretraining-style text. Cosmopedia supplies synthetic textbook/story/blog distributions that are useful for a small experimental language model. OpenOrca and OpenHermes add instruction-following behavior, but should not dominate the corpus. Existing sources maintain continuity with the current 1M run.

## Conversion Strategy

### Pretraining-style rows

For long documents:

```text
source_segments = first window / prompt window
target_text = next 128-256 tokens
task = continuation
```

Use sliding windows with stride:

```text
window_tokens = 384-768
target_tokens = 128-256
stride = 256-512
```

### Instruction rows

For instruction/chat datasets:

```text
source_segments = instruction + optional context + prior turns
target_text = assistant response
task = response generation
```

### Summary / QA rows

For QA and summarization:

```text
source_segments = source document/context
query = question or summarize instruction
target_text = answer or summary
```

## Disk Strategy

Use shard-level rolling training for 10M. Do not materialize all graph JSONL on a 200G disk at once.

Recommended shard size:

```text
raw shard: 25k-50k records
graph shard: 10k-25k records, depending on graph expansion size
queue buffer: 3-5 shards
active training shards: 1 shard per worker partition
delete policy: delete shard only after optimizer step/checkpoint boundary confirms consumption
```

## Priority Order

1. Build a 500k mixed pilot from FineWeb-Edu + Cosmopedia + OpenOrca.
2. Convert to token graph shards.
3. Train 5k-10k steps and compare `lm_loss`, sample quality, and token attribution.
4. If stable, scale to 10M with rolling shard deletion.

## Risks

- ODC-By datasets require attribution in released artifacts.
- Some instruction datasets are synthetic-distillation datasets; keep source proportions explicit.
- Web corpora may include noisy boilerplate. Apply length, language, repetition, and quality filters before graph building.
- 10M records will exceed current disk if fully materialized as graph JSONL. Rolling shard training is required.

