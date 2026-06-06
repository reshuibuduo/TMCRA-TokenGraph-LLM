# Token-Level Semantic Graph Schema

This schema is for training-time graph supervision in TGCLM. It keeps the base
token graph deterministic while allowing an LLM teacher or a rule-free
preprocessor to add token-level semantic edges.

## Goal

TGCLM should learn language reasoning as graph dynamics:

- token nodes carry the language surface;
- typed candidate edges expose possible relations;
- the graph model learns edge gates, path participation, tunnel links, and the
  next-token distribution;
- target text is used only as `target_ids`, not as input evidence.

## Input Fields

Rows may contain the normal schema2 fields:

```json
{
  "sample_id": "row id",
  "query": "prompt or question",
  "source_segments": [
    {"segment_id": "seg1", "text": "source text"}
  ],
  "text_units": [
    {"unit_id": "u1", "text": "optional unit text"}
  ],
  "target_text": "next text to generate"
}
```

Rows may additionally contain `semantic_spans` and `semantic_edges`:

```json
{
  "semantic_spans": [
    {
      "span_id": "cause_1",
      "segment_id": "seg1",
      "quote": "exact quote from the segment",
      "role": "cause"
    },
    {
      "span_id": "effect_1",
      "segment_id": "seg1",
      "quote": "exact quote from the segment",
      "role": "effect"
    }
  ],
  "semantic_edges": [
    {
      "src_span_id": "cause_1",
      "dst_span_id": "effect_1",
      "edge_type": "cause_effect",
      "label": 1
    }
  ]
}
```

Accepted span reference fields:

- `segment_id`
- `segment_ref`
- `source_id`
- `source_segment_id`
- `unit_id`
- `text_unit_id`
- `parent_segment_id`

Accepted span text fields:

- `quote`
- `text`
- `surface`

Accepted direct token boundary fields:

- `token_start`
- `token_end`

## Edge Types

Semantic relation names are normalized into these graph edge types:

- `semantic_same_entity`
- `semantic_entity_attribute`
- `semantic_relation`
- `semantic_cause_effect`
- `semantic_condition_result`
- `semantic_temporal`
- `semantic_definition`
- `semantic_example`
- `semantic_contrast`
- `semantic_part_whole`
- `semantic_quantity`
- `semantic_coreference`
- `semantic_support`
- `semantic_negative`
- `semantic_tunnel`

The builder maps common aliases such as `causal`, `definition`, `example_of`,
`numeric`, `direct_evidence`, `distractor`, and `long_range` into this fixed
edge vocabulary.

## LLM Teacher Role

The LLM teacher should not answer the sample. It should only extract:

- meaningful token spans;
- relation type between spans;
- whether a span is support/evidence or negative/distractor;
- long-range links that the base token graph cannot discover from surface
  overlap.

The teacher output is training supervision. At inference time the graph model
must operate without the teacher.

## Current Builder Behavior

`build_native_token_reasoning_graph_dataset_v3.py` now:

- builds the base token graph from query/context/unit/knowledge tokens;
- deduplicates repeated source/unit text into shared token nodes;
- maps `semantic_spans` to existing token nodes;
- adds typed semantic token-level edges;
- marks semantic support spans as support labels;
- records `semantic_stats` per sample;
- writes the expanded `edge_type_vocab` into `manifest.json`.

## Recommended Use

1. Build the base 1M corpus as usual for language fluency and local transition
   learning.
2. Create a 3k teacher graph smoke set with LLM-assisted semantic spans and
   edges.
3. Train a short run and compare token attribution:
   - do generated tokens bind to semantic edges;
   - do causal/definition/temporal relations appear in top edges;
   - does generation become less local-token-fragment driven.
4. If the smoke test is positive, expand to 30k-100k teacher-supervised rows.

