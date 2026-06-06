# Model Checkpoints

This source package does not include trained checkpoints by default.

Recommended release pattern:

1. Keep source code in Git.
2. Publish checkpoints separately as release assets or on Hugging Face.
3. If storing checkpoints in the repository, use Git LFS.

Current Stage C checkpoint filename:

```text
token_graph_dynamic_decoder_v3.pt
```

Legacy v0.1 checkpoint filename:

```text
token_graph_llm_v1.pt
```

The Stage C checkpoint stores:

```text
model: PyTorch state dict
manifest: graph dataset metadata
args: architecture and training arguments
```

The tokenizer is loaded from the packaged dataset metadata, usually:

```text
tokenizer.json
```

Recommended release asset for Stage C:

```text
tmcra_tokengraph_stagec_model_package_20260606.zip
```

The source repository excludes `.pt`, `.pth`, `.safetensors`, and raw corpora by default.
