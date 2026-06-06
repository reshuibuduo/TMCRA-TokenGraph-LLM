# Release Notes

## v0.2.0-stagec 2026-06-06

This release adds the Stage C / Dynamic Token Graph Decoder V3 line.

Included:

- Stage C source files:
  - `model_token_graph_dynamic_decoder_v3.py`
  - `train_token_graph_dynamic_decoder_v3.py`
  - `train_graph_causal_decoder_v2.py`
  - v3 benchmark/evaluation helpers
- Full-chain training source update:
  - open-corpus schema2 conversion scripts
  - optional semantic teacher annotation scripts
  - token graph dataset builders
  - Stage C full-chain and sharded training templates
- Stage C technical documentation in English and Chinese.
- Full-chain training documentation in English and Chinese.
- Token-level semantic graph schema.
- Stage C smoke benchmark notes:
  - Stage A/B/C loss comparison
  - no_edges / shuffle_edges graph ablation
  - TinyStories validation smoke
  - BLiMP likelihood smoke

Stage C checkpoint package:

- GitHub Release asset: `tmcra_tokengraph_stagec_model_package_20260606.zip`
- checkpoint: `token_graph_dynamic_decoder_v3.pt`
- model size: about 114.6M parameters
- architecture: `dim=512`, `graph_layers=8`, `decoder_layers=10`, untied embeddings
- Release assets are model packages only. Algorithm, builder, trainer, and evaluation code are updated in the source repository.

Current experimental conclusion:

- The Stage C model is substantially stronger than the v0.1 prototype.
- Graph edges materially affect generation: removing or shuffling edges hurts loss and changes output.
- The model can generate early English story-style continuations.
- It is still not a reliable factual QA model or production LLM.

The previous `v0.1.0-prototype` release is now treated as a legacy small prototype checkpoint package.

## 2026-05-30 Prototype Package

This package contains the current TMCRA TokenGraph-LLM research prototype.

Included:

- graph-native autoregressive language model code
- token-level graph dataset builders
- training and checkpoint continuation script
- generalization probe
- token attribution script and HTML output support
- English and Chinese project documentation

Excluded:

- trained checkpoints
- raw/open training corpora
- internal run logs
- private server configuration
- API keys or credentials

Current experimental conclusion:

- The architecture is trainable.
- Loss decreases under next-token and graph-path objectives.
- Token-level attribution can expose top graph nodes and edges for generated tokens.
- Current model quality is early prototype level and not a usable general-purpose LLM.
