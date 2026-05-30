# Release Notes

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
