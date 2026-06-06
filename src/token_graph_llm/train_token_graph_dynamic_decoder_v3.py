from __future__ import annotations

import argparse
import contextlib
import functools
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from model_token_graph_dynamic_decoder_v3 import TokenGraphDynamicCausalLMV3, graph_lm_loss_v3
from native_token_graph_common import BOS, EOS, PAD, UNK, LearnedBpeTokenizer
from train_graph_causal_decoder_v2 import (
    NativeTokenGraphDataset,
    NativeTokenGraphIterableDataset,
    PreloadedNativeTokenGraphDataset,
    collate,
    jsonable_args,
    move_batch,
    resolve_split_paths,
)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def evaluate(
    model: TokenGraphDynamicCausalLMV3,
    loader: DataLoader,
    *,
    pad_id: int,
    device: torch.device,
    max_batches: int,
    label_smoothing: float,
    graph_state_weight: float,
    next_token_node_weight: float,
    edge_type_weight: float,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            moved = move_batch(batch, device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if device.type == "cuda" and amp_dtype is not None
                else contextlib.nullcontext()
            )
            with amp_context:
                losses = graph_lm_loss_v3(
                    model(moved),
                    moved,
                    pad_id=pad_id,
                    label_smoothing=label_smoothing,
                    graph_state_weight=graph_state_weight,
                    next_token_node_weight=next_token_node_weight,
                    edge_type_weight=edge_type_weight,
                )
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            count += 1
    return {key: round(value / max(1, count), 6) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=128)
    parser.add_argument("--max-indexed-lines-per-shard", type=int, default=0)
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--preload-train", action="store_true")
    parser.add_argument("--preload-val", action="store_true")
    parser.add_argument("--amp", choices=["off", "bf16"], default="off")
    parser.add_argument("--max-nodes", type=int, default=384)
    parser.add_argument("--max-edges", type=int, default=1536)
    parser.add_argument("--max-answer-tokens", type=int, default=160)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--graph-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--untie-embeddings", action="store_true")
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--graph-state-weight", type=float, default=0.35)
    parser.add_argument("--next-token-node-weight", type=float, default=0.08)
    parser.add_argument("--edge-type-weight", type=float, default=0.05)
    parser.add_argument("--graph-prior-init", type=float, default=0.0)
    parser.add_argument("--graph-prior-max", type=float, default=1.0)
    parser.add_argument("--prefix-window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=16)
    parser.add_argument("--generate-samples", type=int, default=4)
    parser.add_argument("--generate-temperature", type=float, default=0.8)
    parser.add_argument("--generate-top-k", type=int, default=40)
    parser.add_argument("--generate-repetition-penalty", type=float, default=1.12)
    parser.add_argument("--generate-no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = LearnedBpeTokenizer.from_json(json.loads((args.dataset_dir / "tokenizer.json").read_text(encoding="utf-8")))
    pad_id = int(tokenizer.vocab[PAD])
    bos_id = int(tokenizer.vocab[BOS])
    eos_id = int(tokenizer.vocab[EOS])
    unk_id = int(tokenizer.vocab.get(UNK, 3))
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    node_type_count = max(int(v) for v in manifest["node_type_vocab"].values()) + 1
    edge_type_count = max(int(v) for v in manifest["edge_type_vocab"].values()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None

    train_paths = resolve_split_paths(args.dataset_dir, manifest, "train", "train.jsonl", "train.base.jsonl")
    if args.streaming_train:
        if args.preload_train:
            raise ValueError("--preload-train cannot be combined with --streaming-train")
        train_ds = NativeTokenGraphIterableDataset(
            train_paths,
            limit=args.train_limit,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_answer_tokens=args.max_answer_tokens,
            seed=args.seed,
        )
    else:
        train_dataset_cls = PreloadedNativeTokenGraphDataset if args.preload_train else NativeTokenGraphDataset
        train_ds = train_dataset_cls(
            train_paths,
            limit=args.train_limit,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_answer_tokens=args.max_answer_tokens,
            seed=args.seed,
            max_indexed_lines_per_shard=args.max_indexed_lines_per_shard,
        )
    val_dataset_cls = PreloadedNativeTokenGraphDataset if args.preload_val else NativeTokenGraphDataset
    val_ds = val_dataset_cls(
        resolve_split_paths(args.dataset_dir, manifest, "val", "val.jsonl", "val.base.jsonl"),
        limit=args.val_limit,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_answer_tokens=args.max_answer_tokens,
        seed=args.seed,
        max_indexed_lines_per_shard=args.max_indexed_lines_per_shard,
    )
    collate_fn = functools.partial(collate, pad_id=pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not args.streaming_train,
        num_workers=args.dataloader_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=args.dataloader_workers > 0,
        prefetch_factor=2 if args.dataloader_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=args.dataloader_workers > 0,
        prefetch_factor=2 if args.dataloader_workers > 0 else None,
    )
    model = TokenGraphDynamicCausalLMV3(
        vocab_size=len(tokenizer.vocab),
        node_type_count=node_type_count,
        edge_type_count=edge_type_count,
        dim=args.dim,
        graph_layers=args.graph_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        max_answer_tokens=args.max_answer_tokens,
        dropout=args.dropout,
        tie_embeddings=bool(args.tie_embeddings and not args.untie_embeddings),
        graph_prior_init=args.graph_prior_init,
        graph_prior_max=args.graph_prior_max,
        prefix_window=args.prefix_window,
    ).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(json.dumps({"event": "init_checkpoint_loaded", "missing": missing, "unexpected": unexpected}, ensure_ascii=False), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_stats = count_parameters(model)
    print(json.dumps({"event": "model_init", "device": str(device), "parameters": model_stats, "args": jsonable_args(args)}, ensure_ascii=False), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history: list[dict[str, Any]] = []
    step = 0
    micro_step = 0
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            micro_step += 1
            moved = move_batch(batch, device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if device.type == "cuda" and amp_dtype is not None
                else contextlib.nullcontext()
            )
            with amp_context:
                losses = graph_lm_loss_v3(
                    model(moved),
                    moved,
                    pad_id=pad_id,
                    label_smoothing=args.label_smoothing,
                    graph_state_weight=args.graph_state_weight,
                    next_token_node_weight=args.next_token_node_weight,
                    edge_type_weight=args.edge_type_weight,
                )
            (losses["loss"] / grad_accum_steps).backward()
            if micro_step % grad_accum_steps != 0:
                continue
            step += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if args.log_every and step % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": step, "train_loss": float(losses["loss"].detach().cpu())}, ensure_ascii=False), flush=True)
            if step == 1 or (args.eval_every and step % args.eval_every == 0):
                entry = {"epoch": epoch, "step": step, "train_loss": float(losses["loss"].detach().cpu())}
                entry["val"] = evaluate(
                    model,
                    val_loader,
                    pad_id=pad_id,
                    device=device,
                    max_batches=args.max_val_batches,
                    label_smoothing=args.label_smoothing,
                    graph_state_weight=args.graph_state_weight,
                    next_token_node_weight=args.next_token_node_weight,
                    edge_type_weight=args.edge_type_weight,
                    amp_dtype=amp_dtype,
                )
                history.append(entry)
                print(json.dumps(entry, ensure_ascii=False), flush=True)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    torch.save({"model": model.state_dict(), "manifest": manifest, "args": jsonable_args(args), "parameters": model_stats}, args.out_dir / "token_graph_dynamic_decoder_v3.pt")
    generated: list[dict[str, str]] = []
    sample_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for idx, batch in enumerate(sample_loader):
        if idx >= args.generate_samples:
            break
        moved = move_batch(batch, device)
        ids = model.generate(
            moved,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            unk_id=unk_id,
            max_new_tokens=80,
            min_new_tokens=3,
            temperature=args.generate_temperature,
            top_k=args.generate_top_k,
            repetition_penalty=args.generate_repetition_penalty,
            no_repeat_ngram_size=args.generate_no_repeat_ngram_size,
        )[0].detach().cpu().tolist()
        generated.append({"query": batch["queries"][0], "gold": batch["answers"][0], "pred": tokenizer.decode(ids)})
    summary = {
        "status": "completed",
        "device": str(device),
        "parameters": model_stats,
        "args": jsonable_args(args),
        "history": history,
        "generated": generated,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
