# Model Checkpoints

This source package does not include trained checkpoints by default.

Recommended release pattern:

1. Keep source code in Git.
2. Publish checkpoints separately as release assets or on Hugging Face.
3. If storing checkpoints in the repository, use Git LFS.

Expected checkpoint filename:

```text
token_graph_llm_v1.pt
```

The checkpoint stores:

```text
model: PyTorch state dict
manifest: graph dataset metadata
args: architecture and training arguments
```

The tokenizer is loaded from the dataset directory, usually:

```text
tokenizer.json
```
