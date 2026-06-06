# Package Manifest

Package path:

```text
paper_preparation/token_graph_llm_open_source_20260530
```

## Included

- Source code for the legacy TokenGraph-LLM v1 model.
- Source code for the Stage C / Dynamic Token Graph Decoder V3 model.
- Training and finetuning scripts for v1 and Stage C.
- Generalization probe.
- Token attribution script with JSON and HTML outputs.
- Stage C graph ablation and benchmark smoke scripts.
- Dataset builders for token-level graph training data.
- Hugging Face open-corpus download helper.
- English and Chinese README files.
- Stage C architecture and experiment notes.
- Token-level semantic graph schema.
- MIT license.

## Excluded

- Raw datasets.
- Built train/validation JSONL datasets except the small probe prompt file.
- Checkpoints and other large model artifacts.
- Run logs, PID files, caches, and local virtual environments.
- Internal server configuration and credentials.

## Current Recommended Checkpoint Handling

Keep `.pt` checkpoints outside the source repository by default. Publish them separately as:

- a GitHub release asset, or
- a Hugging Face model artifact, or
- a Git LFS tracked file under `models/`.

Current Stage C checkpoint filename:

```text
token_graph_dynamic_decoder_v3.pt
```

Legacy v0.1 checkpoint filename:

```text
token_graph_llm_v1.pt
```

## Validation Performed

- Python source files compiled successfully with `py_compile`.
- `__pycache__` files were removed.
- No large model artifacts are included.
- Secret scan found no API keys, internal hostnames, local machine paths, or private remote paths.
