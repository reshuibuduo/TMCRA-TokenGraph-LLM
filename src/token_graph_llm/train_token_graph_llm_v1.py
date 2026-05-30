from __future__ import annotations

import argparse
import functools
import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from native_token_graph_common import BOS, EOS, PAD, UNK, LearnedBpeTokenizer
from token_graph_llm_model_v1 import TokenGraphLanguageModel, token_graph_lm_loss


def resolve_split_path(dataset_dir: Path, primary: str, fallback: str) -> Path:
    first = dataset_dir / primary
    return first if first.exists() else dataset_dir / fallback


def resolve_split_paths(dataset_dir: Path, manifest: dict[str, Any], split: str, primary: str, fallback: str) -> list[Path]:
    shard_key = f"{split}_shards"
    shards = manifest.get(shard_key)
    if isinstance(shards, list) and shards:
        return [Path(item) for item in shards]
    return [resolve_split_path(dataset_dir, primary, fallback)]


def row_to_sample(row: dict[str, Any], *, max_nodes: int, max_edges: int, max_sequence_tokens: int) -> dict[str, Any]:
    nodes = list(row.get("nodes", []))[:max_nodes]
    edges = [
        edge
        for edge in row.get("edges", [])
        if int(edge.get("src", 0)) < len(nodes) and int(edge.get("dst", 0)) < len(nodes)
    ][:max_edges]
    sequence_ids = list(row.get("target_ids", []) or row.get("answer_ids", []) or [])[:max_sequence_tokens]
    return {
        "prompt": row.get("query", "") or row.get("prompt", ""),
        "target_text": row.get("target_text", "") or row.get("answer", ""),
        "node_token_ids": [int(node.get("piece_id", 3)) for node in nodes],
        "node_types": [int(node.get("node_type_id", 0)) for node in nodes],
        "sequence_ids": sequence_ids,
        "edges": edges,
    }


class TokenGraphDataset(Dataset):
    def __init__(
        self,
        paths: Path | list[Path],
        *,
        limit: int,
        max_nodes: int,
        max_edges: int,
        max_sequence_tokens: int,
        seed: int,
        max_indexed_lines_per_shard: int = 0,
    ) -> None:
        self.paths = [paths] if isinstance(paths, Path) else list(paths)
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_sequence_tokens = max_sequence_tokens
        offsets: list[tuple[Path, int]] = []
        for path in self.paths:
            with path.open("rb") as fh:
                pos = fh.tell()
                indexed = 0
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    offsets.append((path, pos))
                    indexed += 1
                    if max_indexed_lines_per_shard and indexed >= max_indexed_lines_per_shard:
                        break
                    pos = fh.tell()
        rng = random.Random(seed)
        rng.shuffle(offsets)
        self.offsets = offsets[:limit] if limit else offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, pos = self.offsets[idx]
        with path.open("rb") as fh:
            fh.seek(pos)
            row = json.loads(fh.readline().decode("utf-8"))
        return row_to_sample(row, max_nodes=self.max_nodes, max_edges=self.max_edges, max_sequence_tokens=self.max_sequence_tokens)


class TokenGraphIterableDataset(IterableDataset):
    def __init__(
        self,
        paths: list[Path],
        *,
        limit: int,
        max_nodes: int,
        max_edges: int,
        max_sequence_tokens: int,
        seed: int,
    ) -> None:
        self.paths = list(paths)
        self.limit = limit
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_sequence_tokens = max_sequence_tokens
        self.seed = seed

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        paths = list(self.paths)
        random.Random(self.seed).shuffle(paths)
        if worker is not None:
            paths = paths[worker.id :: worker.num_workers]
        emitted = 0
        for path in paths:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if self.limit and emitted >= self.limit:
                        return
                    if not line.strip():
                        continue
                    emitted += 1
                    row = json.loads(line)
                    yield row_to_sample(
                        row,
                        max_nodes=self.max_nodes,
                        max_edges=self.max_edges,
                        max_sequence_tokens=self.max_sequence_tokens,
                    )


def pad_1d(rows: list[list[int]], *, fill: int) -> torch.Tensor:
    width = max(1, max(len(row) for row in rows))
    out = torch.full((len(rows), width), fill, dtype=torch.long)
    for index, row in enumerate(rows):
        if row:
            out[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def collate(rows: list[dict[str, Any]], *, pad_id: int, node_type_pad: int = 0, edge_type_pad: int = 0) -> dict[str, Any]:
    node_token_ids = pad_1d([r["node_token_ids"] for r in rows], fill=pad_id)
    node_types = pad_1d([r["node_types"] for r in rows], fill=node_type_pad)
    node_mask = torch.zeros_like(node_token_ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        node_mask[index, : len(row["node_token_ids"])] = True
    sequence_ids = pad_1d([r["sequence_ids"] for r in rows], fill=pad_id)
    max_edges = max(1, max(len(r["edges"]) for r in rows))
    edge_src = torch.zeros((len(rows), max_edges), dtype=torch.long)
    edge_dst = torch.zeros((len(rows), max_edges), dtype=torch.long)
    edge_types = torch.full((len(rows), max_edges), edge_type_pad, dtype=torch.long)
    edge_mask = torch.zeros((len(rows), max_edges), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        for edge_index, edge in enumerate(row["edges"]):
            edge_src[row_index, edge_index] = int(edge.get("src", 0))
            edge_dst[row_index, edge_index] = int(edge.get("dst", 0))
            edge_types[row_index, edge_index] = int(edge.get("edge_type_id", edge_type_pad))
            edge_mask[row_index, edge_index] = True
    return {
        "prompts": [r["prompt"] for r in rows],
        "target_texts": [r["target_text"] for r in rows],
        "node_token_ids": node_token_ids,
        "node_types": node_types,
        "node_mask": node_mask,
        "sequence_ids": sequence_ids,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_types": edge_types,
        "edge_mask": edge_mask,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def evaluate(
    model: TokenGraphLanguageModel,
    loader: DataLoader,
    *,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    max_batches: int,
    label_smoothing: float,
    token_path_weight: float,
    transition_path_weight: float,
    relation_transition_weight: float,
    causal_path_weight: float,
    non_eos_weight: float,
    non_eos_steps: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            moved = move_batch(batch, device)
            losses = token_graph_lm_loss(
                model(moved),
                moved,
                pad_id=pad_id,
                eos_id=eos_id,
                label_smoothing=label_smoothing,
                token_path_weight=token_path_weight,
                transition_path_weight=transition_path_weight,
                relation_transition_weight=relation_transition_weight,
                causal_path_weight=causal_path_weight,
                non_eos_weight=non_eos_weight,
                non_eos_steps=non_eos_steps,
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
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--max-indexed-lines-per-shard", type=int, default=0)
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=384)
    parser.add_argument("--max-edges", type=int, default=1536)
    parser.add_argument("--max-sequence-tokens", type=int, default=160)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--graph-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--untie-embeddings", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--token-path-weight", type=float, default=0.0)
    parser.add_argument("--transition-path-weight", type=float, default=0.0)
    parser.add_argument("--relation-transition-weight", type=float, default=0.0)
    parser.add_argument("--causal-path-weight", type=float, default=0.0)
    parser.add_argument("--non-eos-weight", type=float, default=0.0)
    parser.add_argument("--non-eos-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=8)
    parser.add_argument("--generate-samples", type=int, default=4)
    parser.add_argument("--generate-temperature", type=float, default=0.85)
    parser.add_argument("--generate-top-k", type=int, default=40)
    parser.add_argument("--generate-repetition-penalty", type=float, default=1.15)
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

    train_paths = resolve_split_paths(args.dataset_dir, manifest, "train", "train.jsonl", "train.base.jsonl")
    if args.streaming_train:
        train_ds = TokenGraphIterableDataset(
            train_paths,
            limit=args.train_limit,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_sequence_tokens=args.max_sequence_tokens,
            seed=args.seed,
        )
    else:
        train_ds = TokenGraphDataset(
            train_paths,
            limit=args.train_limit,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_sequence_tokens=args.max_sequence_tokens,
            seed=args.seed,
            max_indexed_lines_per_shard=args.max_indexed_lines_per_shard,
        )
    val_ds = TokenGraphDataset(
        resolve_split_paths(args.dataset_dir, manifest, "val", "val.jsonl", "val.base.jsonl"),
        limit=args.val_limit,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_sequence_tokens=args.max_sequence_tokens,
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
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = TokenGraphLanguageModel(
        vocab_size=len(tokenizer.vocab),
        node_type_count=node_type_count,
        edge_type_count=edge_type_count,
        dim=args.dim,
        graph_layers=args.graph_layers,
        decoder_layers=args.decoder_layers,
        max_sequence_tokens=args.max_sequence_tokens,
        dropout=args.dropout,
        tie_embeddings=not args.untie_embeddings,
    ).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                json.dumps(
                    {"event": "init_checkpoint_loaded", "missing": missing, "unexpected": unexpected},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    step = 0
    micro_step = 0
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            micro_step += 1
            moved = move_batch(batch, device)
            losses = token_graph_lm_loss(
                model(moved),
                moved,
                pad_id=pad_id,
                eos_id=eos_id,
                label_smoothing=args.label_smoothing,
                token_path_weight=args.token_path_weight,
                transition_path_weight=args.transition_path_weight,
                relation_transition_weight=args.relation_transition_weight,
                causal_path_weight=args.causal_path_weight,
                non_eos_weight=args.non_eos_weight,
                non_eos_steps=args.non_eos_steps,
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
                    eos_id=eos_id,
                    device=device,
                    max_batches=args.max_val_batches,
                    label_smoothing=args.label_smoothing,
                    token_path_weight=args.token_path_weight,
                    transition_path_weight=args.transition_path_weight,
                    relation_transition_weight=args.relation_transition_weight,
                    causal_path_weight=args.causal_path_weight,
                    non_eos_weight=args.non_eos_weight,
                    non_eos_steps=args.non_eos_steps,
                )
                history.append(entry)
                print(json.dumps(entry, ensure_ascii=False), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    torch.save({"model": model.state_dict(), "manifest": manifest, "args": jsonable_args(args)}, args.out_dir / "token_graph_llm_v1.pt")
    generated: list[dict[str, str]] = []
    sample_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for index, batch in enumerate(sample_loader):
        if index >= args.generate_samples:
            break
        moved = move_batch(batch, device)
        ids = model.generate(
            moved,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            unk_id=unk_id,
            max_new_tokens=64,
            min_new_tokens=3,
            temperature=args.generate_temperature,
            top_k=args.generate_top_k,
            repetition_penalty=args.generate_repetition_penalty,
            no_repeat_ngram_size=args.generate_no_repeat_ngram_size,
        )[0].detach().cpu().tolist()
        generated.append({"prompt": batch["prompts"][0], "gold": batch["target_texts"][0], "pred": tokenizer.decode(ids)})
    summary = {"status": "completed", "device": str(device), "args": jsonable_args(args), "history": history, "generated": generated}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
