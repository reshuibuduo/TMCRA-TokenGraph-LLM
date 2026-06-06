from __future__ import annotations

import argparse
import functools
import json
import random
from pathlib import Path
from typing import Any

import itertools

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from model_graph_causal_decoder_v2 import TokenGraphCausalLM, graph_lm_loss
from native_token_graph_common import BOS, EOS, PAD, UNK, LearnedBpeTokenizer, load_jsonl


def row_to_sample(row: dict[str, Any], *, max_nodes: int, max_edges: int, max_answer_tokens: int) -> dict[str, Any]:
    nodes = list(row.get("nodes", []) or [])[:max_nodes]
    local_count = len(nodes)
    target_prefix_positions = [
        int(node.get("position", -1)) if int(node.get("node_type_id", 0)) == 11 or str(node.get("node_type", "")) == "target_prefix_token" else -1
        for node in nodes
    ]
    valid_edges = []
    for edge in row.get("edges", []) or []:
        src = int(edge.get("src", -1))
        dst = int(edge.get("dst", -1))
        if 0 <= src < local_count and 0 <= dst < local_count:
            valid_edges.append(edge)
        if len(valid_edges) >= max_edges:
            break
    answer_ids = list(row.get("target_ids", []) or row.get("answer_ids", []) or [])[:max_answer_tokens]
    return {
        "sample_id": row.get("sample_id", ""),
        "query": row.get("query", ""),
        "answer": row.get("target_text", "") or row.get("answer", ""),
        "node_token_ids": [int(node.get("piece_id", 3)) for node in nodes],
        "node_types": [int(node.get("node_type_id", 0)) for node in nodes],
        "support_labels": [float(node.get("support_label", 0)) for node in nodes],
        "answer_overlap_labels": [float(node.get("answer_overlap_label", 0)) for node in nodes],
        "target_prefix_positions": target_prefix_positions,
        "edges": valid_edges,
        "answer_ids": answer_ids,
    }


class NativeTokenGraphDataset(Dataset):
    def __init__(
        self,
        paths: Path | list[Path],
        *,
        limit: int,
        max_nodes: int,
        max_edges: int,
        max_answer_tokens: int,
        seed: int,
        max_indexed_lines_per_shard: int = 0,
    ) -> None:
        self.paths = [paths] if isinstance(paths, Path) else list(paths)
        offsets: list[tuple[int, int]] = []
        for path_index, path in enumerate(self.paths):
            indexed_in_shard = 0
            with path.open("rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append((path_index, offset))
                        indexed_in_shard += 1
                        if max_indexed_lines_per_shard and indexed_in_shard >= max_indexed_lines_per_shard:
                            break
        rng = random.Random(seed)
        rng.shuffle(offsets)
        self.offsets = offsets[:limit] if limit else offsets
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_answer_tokens = max_answer_tokens
        self._files: dict[int, Any] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_files"] = {}
        return state

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_row(self, idx: int) -> dict[str, Any]:
        path_index, offset = self.offsets[idx]
        if path_index not in self._files:
            self._files[path_index] = self.paths[path_index].open("rb")
        file_obj = self._files[path_index]
        file_obj.seek(offset)
        return json.loads(file_obj.readline().decode("utf-8"))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._read_row(idx)
        return row_to_sample(
            row,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            max_answer_tokens=self.max_answer_tokens,
        )


class PreloadedNativeTokenGraphDataset(Dataset):
    def __init__(
        self,
        paths: Path | list[Path],
        *,
        limit: int,
        max_nodes: int,
        max_edges: int,
        max_answer_tokens: int,
        seed: int,
        max_indexed_lines_per_shard: int = 0,
    ) -> None:
        self.paths = [paths] if isinstance(paths, Path) else list(paths)
        samples: list[dict[str, Any]] = []
        for path in self.paths:
            indexed_in_shard = 0
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    samples.append(
                        row_to_sample(
                            json.loads(line),
                            max_nodes=max_nodes,
                            max_edges=max_edges,
                            max_answer_tokens=max_answer_tokens,
                        )
                    )
                    indexed_in_shard += 1
                    if max_indexed_lines_per_shard and indexed_in_shard >= max_indexed_lines_per_shard:
                        break
        rng = random.Random(seed)
        rng.shuffle(samples)
        self.samples = samples[:limit] if limit else samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class NativeTokenGraphIterableDataset(IterableDataset):
    def __init__(
        self,
        paths: list[Path],
        *,
        limit: int,
        max_nodes: int,
        max_edges: int,
        max_answer_tokens: int,
        seed: int,
    ) -> None:
        self.paths = list(paths)
        self.limit = limit
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_answer_tokens = max_answer_tokens
        self.seed = seed

    def __iter__(self):
        worker = get_worker_info()
        paths = list(self.paths)
        rng = random.Random(self.seed + (worker.id if worker else 0))
        rng.shuffle(paths)
        if worker is not None:
            paths = paths[worker.id :: worker.num_workers]

        emitted = 0
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    yield row_to_sample(
                        row,
                        max_nodes=self.max_nodes,
                        max_edges=self.max_edges,
                        max_answer_tokens=self.max_answer_tokens,
                    )
                    emitted += 1
                    if self.limit and emitted >= self.limit:
                        return


def pad_1d(values: list[list[int]], *, fill: int) -> Tensor:
    width = max(1, max(len(row) for row in values))
    out = torch.full((len(values), width), fill, dtype=torch.long)
    for i, row in enumerate(values):
        if row:
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def pad_1d_float(values: list[list[float]], *, fill: float) -> Tensor:
    width = max(1, max(len(row) for row in values))
    out = torch.full((len(values), width), fill, dtype=torch.float32)
    for i, row in enumerate(values):
        if row:
            out[i, : len(row)] = torch.tensor(row, dtype=torch.float32)
    return out


def collate(rows: list[dict[str, Any]], *, pad_id: int, node_type_pad: int = 0, edge_type_pad: int = 0) -> dict[str, Any]:
    node_token_ids = pad_1d([r["node_token_ids"] for r in rows], fill=pad_id)
    node_types = pad_1d([r["node_types"] for r in rows], fill=node_type_pad)
    node_mask = torch.zeros_like(node_token_ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        node_mask[i, : len(row["node_token_ids"])] = True
    target_prefix_positions = pad_1d([r["target_prefix_positions"] for r in rows], fill=-1)
    full_node_mask = node_mask.clone()
    # Target-prefix nodes are teacher-forcing nodes. A static graph encoder would
    # let future target tokens leak through ordinary context nodes, so the safe
    # default is to remove these nodes from the encoder-side context graph.
    node_mask = node_mask & target_prefix_positions.lt(0)
    support_labels = pad_1d_float([r["support_labels"] for r in rows], fill=0.0)
    answer_overlap_labels = pad_1d_float([r["answer_overlap_labels"] for r in rows], fill=0.0)
    answer_ids = pad_1d([r["answer_ids"] for r in rows], fill=pad_id)
    max_edges = max(1, max(len(r["edges"]) for r in rows))
    edge_src = torch.zeros((len(rows), max_edges), dtype=torch.long)
    edge_dst = torch.zeros((len(rows), max_edges), dtype=torch.long)
    edge_types = torch.full((len(rows), max_edges), edge_type_pad, dtype=torch.long)
    edge_mask = torch.zeros((len(rows), max_edges), dtype=torch.bool)
    for i, row in enumerate(rows):
        for j, edge in enumerate(row["edges"]):
            edge_src[i, j] = int(edge.get("src", 0))
            edge_dst[i, j] = int(edge.get("dst", 0))
            edge_types[i, j] = int(edge.get("edge_type_id", edge_type_pad))
            edge_mask[i, j] = True
    if edge_mask.any():
        edge_src_visible = torch.gather(node_mask, 1, edge_src.clamp_min(0))
        edge_dst_visible = torch.gather(node_mask, 1, edge_dst.clamp_min(0))
        edge_mask = edge_mask & edge_src_visible & edge_dst_visible
    return {
        "sample_ids": [r["sample_id"] for r in rows],
        "queries": [r["query"] for r in rows],
        "answers": [r["answer"] for r in rows],
        "node_token_ids": node_token_ids,
        "node_types": node_types,
        "node_mask": node_mask,
        "full_node_mask": full_node_mask,
        "target_prefix_positions": target_prefix_positions,
        "support_labels": support_labels,
        "answer_overlap_labels": answer_overlap_labels,
        "answer_ids": answer_ids,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_types": edge_types,
        "edge_mask": edge_mask,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def resolve_split_path(dataset_dir: Path, primary: str, fallback: str) -> Path:
    primary_path = dataset_dir / primary
    if primary_path.exists():
        return primary_path
    fallback_path = dataset_dir / fallback
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(f"Missing split file: {primary_path} or {fallback_path}")


def resolve_split_paths(dataset_dir: Path, manifest: dict[str, Any], split: str, primary: str, fallback: str) -> list[Path]:
    shard_key = f"{split}_shards"
    shards = manifest.get(shard_key)
    if isinstance(shards, list) and shards:
        return [Path(item) for item in shards]
    return [resolve_split_path(dataset_dir, primary, fallback)]


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def evaluate(
    model: TokenGraphCausalLM,
    loader: DataLoader,
    *,
    pad_id: int,
    device: torch.device,
    max_batches: int,
    label_smoothing: float,
    graph_state_weight: float,
    next_token_node_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            moved = move_batch(batch, device)
            losses = graph_lm_loss(
                model(moved),
                moved,
                pad_id=pad_id,
                label_smoothing=label_smoothing,
                graph_state_weight=graph_state_weight,
                next_token_node_weight=next_token_node_weight,
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
    parser.add_argument("--max-answer-tokens", type=int, default=160)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--graph-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--untie-embeddings", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--graph-state-weight", type=float, default=0.0)
    parser.add_argument("--next-token-node-weight", type=float, default=0.0)
    parser.add_argument("--graph-prior-init", type=float, default=0.35)
    parser.add_argument("--graph-prior-max", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=8)
    parser.add_argument("--generate-samples", type=int, default=4)
    parser.add_argument("--generate-temperature", type=float, default=0.85)
    parser.add_argument("--generate-top-k", type=int, default=40)
    parser.add_argument("--generate-base-top-k", type=int, default=0)
    parser.add_argument("--generate-graph-vocab-top-k", type=int, default=0)
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
        train_ds = NativeTokenGraphIterableDataset(
            train_paths,
            limit=args.train_limit,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_answer_tokens=args.max_answer_tokens,
            seed=args.seed,
        )
    else:
        train_ds = NativeTokenGraphDataset(
            train_paths,
            limit=args.train_limit,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_answer_tokens=args.max_answer_tokens,
            seed=args.seed,
            max_indexed_lines_per_shard=args.max_indexed_lines_per_shard,
        )
    val_ds = NativeTokenGraphDataset(
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
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = TokenGraphCausalLM(
        vocab_size=len(tokenizer.vocab),
        node_type_count=node_type_count,
        edge_type_count=edge_type_count,
        dim=args.dim,
        graph_layers=args.graph_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        max_answer_tokens=args.max_answer_tokens,
        dropout=args.dropout,
        tie_embeddings=not args.untie_embeddings,
        graph_prior_init=args.graph_prior_init,
        graph_prior_max=args.graph_prior_max,
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
            losses = graph_lm_loss(
                model(moved),
                moved,
                pad_id=pad_id,
                label_smoothing=args.label_smoothing,
                graph_state_weight=args.graph_state_weight,
                next_token_node_weight=args.next_token_node_weight,
            )
            (losses["loss"] / grad_accum_steps).backward()
            if micro_step % grad_accum_steps != 0:
                continue
            step += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if args.log_every and step % args.log_every == 0:
                print(
                    json.dumps(
                        {"epoch": epoch, "step": step, "train_loss": float(losses["loss"].detach().cpu())},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
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
                )
                history.append(entry)
                print(json.dumps(entry, ensure_ascii=False), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    torch.save({"model": model.state_dict(), "manifest": manifest, "args": jsonable_args(args)}, args.out_dir / "token_graph_causal_lm_v2.pt")
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
            max_new_tokens=64,
            min_new_tokens=3,
            temperature=args.generate_temperature,
            top_k=args.generate_top_k,
            base_top_k=args.generate_base_top_k,
            graph_vocab_top_k=args.generate_graph_vocab_top_k,
            repetition_penalty=args.generate_repetition_penalty,
            no_repeat_ngram_size=args.generate_no_repeat_ngram_size,
        )[0].detach().cpu().tolist()
        generated.append({"query": batch["queries"][0], "gold": batch["answers"][0], "pred": tokenizer.decode(ids)})
    summary = {"status": "completed", "device": str(device), "args": jsonable_args(args), "history": history, "generated": generated}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
