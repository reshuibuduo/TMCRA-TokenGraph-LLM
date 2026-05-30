# Package Manifest

Package path:

```text
paper_preparation/token_graph_llm_open_source_20260530
```

## Included

- Source code for the TokenGraph-LLM v1 model.
- Training and finetuning script.
- Generalization probe.
- Token attribution script with JSON and HTML outputs.
- Dataset builders for token-level graph training data.
- Hugging Face open-corpus download helper.
- English and Chinese README files.
- Architecture and experiment notes.
- MIT license.

## Excluded

- Raw datasets.
- Built train/validation JSONL datasets except the small probe prompt file.
- Checkpoints and other large model artifacts.
- Run logs, PID files, caches, and local virtual environments.
- Internal server configuration and credentials.

## Current Recommended Checkpoint Handling

Keep `token_graph_llm_v1.pt` outside the source repository by default. Publish it separately as:

- a GitHub release asset, or
- a Hugging Face model artifact, or
- a Git LFS tracked file under `models/`.

## Validation Performed

- Python source files compiled successfully with `py_compile`.
- `__pycache__` files were removed.
- No large model artifacts are included.
- Secret scan found no API keys, internal hostnames, local machine paths, or private remote paths.
