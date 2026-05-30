# Experiment Summary

This summary records the current state of the prototype before open-source packaging.

## Base Run

Configuration:

- dimension: 384
- graph layers: 6
- decoder layers: 8
- untied embeddings: true
- training corpus: about 1M graph samples
- main objective: next-token prediction
- auxiliary objectives:
  - token path loss
  - transition path loss

Observed behavior:

- The model learned English-like token fragments.
- It did not become a usable general LLM.
- Greedy outputs often showed short-answer and `unknown`-like collapse.
- Sampling showed the model was not simply memorizing fixed outputs.

## Relation Finetune

Added objectives:

- relation transition loss
- causal path consistency loss

Observed validation trend during a 3k-step finetune:

```text
val loss:                  1.9668 -> 1.8843
lm loss:                   1.8828 -> 1.8169
relation transition loss:  1.4001 -> 1.2082
causal path loss:          3.1127 -> 1.5699
```

Interpretation:

- The new graph-path objectives have learnable signal.
- Language modeling did not collapse during the 3k-step finetune.
- Generation quality remained weak, especially for factual QA and instruction following.

## Non-EOS Regularization

Added objective:

- non-EOS regularization over early target positions

Purpose:

- Discourage early EOS probability.
- Reduce short-answer collapse without adding task-specific rules.

Initial smoke result:

- The objective runs correctly.
- The measured non-EOS loss was very small, which suggests short-answer collapse is not only an EOS issue.

## Current Capability Boundary

Relatively stronger:

- short English story-style continuation
- local entity/topic reuse
- partial English sentence shape

Weak:

- exact QA
- numbers and factual extraction
- list generation
- abstract definitions
- multilingual prompts
- robust instruction following

Current classification:

> A trainable graph-native language model prototype, not a mature LLM and not a pure memorization system.
