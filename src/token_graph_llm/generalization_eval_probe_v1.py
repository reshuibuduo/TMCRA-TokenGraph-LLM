from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch

from native_token_graph_common import BOS, EOS, PAD, UNK, LearnedBpeTokenizer
from token_graph_llm_model_v1 import TokenGraphLanguageModel
from train_token_graph_llm_v1 import collate, move_batch, resolve_split_paths, row_to_sample


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text or "").lower())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(tokens))


def load_model(run_dir: Path, dataset_dir: Path, device: torch.device) -> tuple[TokenGraphLanguageModel, LearnedBpeTokenizer, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(run_dir / "token_graph_llm_v1.pt", map_location=device, weights_only=False)
    manifest = checkpoint["manifest"]
    args = checkpoint.get("args", {})
    tokenizer = LearnedBpeTokenizer.from_json(json.loads((dataset_dir / "tokenizer.json").read_text(encoding="utf-8")))
    model = TokenGraphLanguageModel(
        vocab_size=len(tokenizer.vocab),
        node_type_count=max(int(v) for v in manifest["node_type_vocab"].values()) + 1,
        edge_type_count=max(int(v) for v in manifest["edge_type_vocab"].values()) + 1,
        dim=int(args.get("dim", 384)),
        graph_layers=int(args.get("graph_layers", 6)),
        decoder_layers=int(args.get("decoder_layers", 8)),
        max_sequence_tokens=int(args.get("max_sequence_tokens", 160)),
        dropout=float(args.get("dropout", 0.0)),
        tie_embeddings=not bool(args.get("untie_embeddings", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, manifest, args


def make_prompt_sample(prompt: str, tokenizer: LearnedBpeTokenizer, manifest: dict[str, Any], *, max_nodes: int, max_edges: int, max_sequence_tokens: int) -> dict[str, Any]:
    node_type_vocab = manifest["node_type_vocab"]
    edge_type_vocab = manifest["edge_type_vocab"]
    pieces = tokenizer.encode_pieces(prompt, max_tokens=max_nodes)
    ids = [tokenizer.vocab.get(piece, tokenizer.vocab.get(UNK, 3)) for piece in pieces]
    nodes = [
        {
            "piece": piece,
            "piece_id": int(token_id),
            "node_type_id": int(node_type_vocab.get("context_token", node_type_vocab.get("token", 0))),
        }
        for piece, token_id in zip(pieces, ids)
    ]
    edges: list[dict[str, Any]] = []
    seq_edge = int(edge_type_vocab.get("sequential_next", edge_type_vocab.get("token_next", 0)))
    for index in range(len(nodes) - 1):
        edges.append({"src": index, "dst": index + 1, "edge_type_id": seq_edge})
    row = {
        "query": prompt,
        "target_text": "",
        "target_ids": [tokenizer.vocab[BOS], tokenizer.vocab[EOS]],
        "nodes": nodes[:max_nodes],
        "edges": edges[:max_edges],
    }
    return row_to_sample(row, max_nodes=max_nodes, max_edges=max_edges, max_sequence_tokens=max_sequence_tokens)


def iter_train_rows(dataset_dir: Path, manifest: dict[str, Any], *, max_rows: int, max_nodes: int, max_edges: int, max_sequence_tokens: int) -> Iterable[dict[str, Any]]:
    paths = resolve_split_paths(dataset_dir, manifest, "train", "train.jsonl", "train.base.jsonl")
    emitted = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if max_rows and emitted >= max_rows:
                    return
                if not line.strip():
                    continue
                row = json.loads(line)
                emitted += 1
                yield row_to_sample(row, max_nodes=max_nodes, max_edges=max_edges, max_sequence_tokens=max_sequence_tokens)


def nearest_train(prompt: str, train_cache: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_words = words(prompt)
    best: dict[str, Any] = {"similarity": 0.0, "prompt": "", "target": ""}
    for row in train_cache:
        sim = jaccard(prompt_words, words(row.get("prompt", "")))
        if sim > best["similarity"]:
            best = {"similarity": sim, "prompt": row.get("prompt", ""), "target": row.get("target_text", "")}
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--prompts-jsonl", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--train-neighbor-scan", type=int, default=50000)
    parser.add_argument("--max-nodes", type=int, default=384)
    parser.add_argument("--max-edges", type=int, default=1536)
    parser.add_argument("--max-sequence-tokens", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, manifest, train_args = load_model(args.run_dir, args.dataset_dir, device)
    pad_id = int(tokenizer.vocab[PAD])
    bos_id = int(tokenizer.vocab[BOS])
    eos_id = int(tokenizer.vocab[EOS])
    unk_id = int(tokenizer.vocab.get(UNK, 3))
    train_cache = list(
        iter_train_rows(
            args.dataset_dir,
            manifest,
            max_rows=args.train_neighbor_scan,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_sequence_tokens=args.max_sequence_tokens,
        )
    )
    probes = [json.loads(line) for line in args.prompts_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs: list[dict[str, Any]] = []
    for probe in probes:
        sample = make_prompt_sample(
            probe["prompt"],
            tokenizer,
            manifest,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_sequence_tokens=args.max_sequence_tokens,
        )
        batch = collate([sample], pad_id=pad_id)
        moved = move_batch(batch, device)
        ids = model.generate(
            moved,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            unk_id=unk_id,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=3,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )[0].detach().cpu().tolist()
        output = tokenizer.decode(ids)
        prompt_words = words(probe["prompt"])
        output_words = words(output)
        nearest = nearest_train(probe["prompt"], train_cache)
        outputs.append(
            {
                **probe,
                "output": output,
                "output_word_count": len(output_words),
                "copy_ratio_prompt_words": round(sum(1 for word in output_words if word in set(prompt_words)) / max(1, len(output_words)), 4),
                "new_token_ratio": round(sum(1 for word in output_words if word not in set(prompt_words)) / max(1, len(output_words)), 4),
                "repetition_ratio": round(repetition_ratio(output_words), 4),
                "nearest_train": {
                    "similarity": round(float(nearest["similarity"]), 4),
                    "prompt": nearest["prompt"][:500],
                    "target": nearest["target"][:500],
                },
            }
        )
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in outputs:
        by_category.setdefault(row["category"], []).append(row)
    summary = {
        "status": "completed",
        "device": str(device),
        "run_dir": str(args.run_dir),
        "dataset_dir": str(args.dataset_dir),
        "train_neighbor_scan": len(train_cache),
        "count": len(outputs),
        "category_summary": {
            category: {
                "count": len(rows),
                "avg_nearest_similarity": round(sum(r["nearest_train"]["similarity"] for r in rows) / max(1, len(rows)), 4),
                "avg_copy_ratio": round(sum(r["copy_ratio_prompt_words"] for r in rows) / max(1, len(rows)), 4),
                "avg_new_token_ratio": round(sum(r["new_token_ratio"] for r in rows) / max(1, len(rows)), 4),
                "avg_repetition_ratio": round(sum(r["repetition_ratio"] for r in rows) / max(1, len(rows)), 4),
            }
            for category, rows in by_category.items()
        },
        "items": outputs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
