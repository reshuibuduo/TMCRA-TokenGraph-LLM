from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from build_native_token_reasoning_graph_dataset_v3 import build_graph
from model_token_graph_dynamic_decoder_v3 import TokenGraphDynamicCausalLMV3
from native_token_graph_common import PAD, LearnedBpeTokenizer
from train_graph_causal_decoder_v2 import collate, move_batch


class PairGraphDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        nodes = list(row.get("nodes", []) or [])
        node_count = len(nodes)
        edges = []
        for edge in row.get("edges", []) or []:
            src = int(edge.get("src", -1))
            dst = int(edge.get("dst", -1))
            if 0 <= src < node_count and 0 <= dst < node_count:
                edges.append(edge)
        return {
            "sample_id": row.get("sample_id", ""),
            "query": row.get("query", ""),
            "answer": row.get("target_text", ""),
            "node_token_ids": [int(node.get("piece_id", 3)) for node in nodes],
            "node_types": [int(node.get("node_type_id", 0)) for node in nodes],
            "support_labels": [float(node.get("support_label", 0.0)) for node in nodes],
            "answer_overlap_labels": [float(node.get("answer_overlap_label", 0.0)) for node in nodes],
            "target_prefix_positions": [
                int(node.get("position", -1))
                if int(node.get("node_type_id", 0)) == 11 or str(node.get("node_type", "")) == "target_prefix_token"
                else -1
                for node in nodes
            ],
            "edges": edges,
            "answer_ids": list(row.get("target_ids", []) or row.get("answer_ids", [])),
        }


def make_sentence_row(sample_id: str, sentence: str) -> dict[str, Any]:
    return {
        "schema_version": "token_graph_corpus_v2",
        "sample_id": sample_id,
        "source": "blimp_likelihood_smoke",
        "split": "probe",
        "query": "Generate a grammatical English sentence.",
        "question_date": None,
        "source_segments": [
            {
                "segment_id": "instruction",
                "text": "Generate a grammatical English sentence.",
                "segment_type": "instruction",
                "timestamp": None,
                "source_port": "blimp",
            }
        ],
        "text_units": [
            {
                "unit_id": "instruction:u1",
                "parent_segment_id": "instruction",
                "text": "Generate a grammatical English sentence.",
                "unit_type": "instruction",
                "numeric_surface": "",
                "temporal_surface": "",
                "state_surface": "",
                "source_segment_type": "instruction",
            }
        ],
        "support_segment_ids": ["instruction"],
        "support_text_unit_ids": ["instruction:u1"],
        "support_alignment": [],
        "target_text": sentence,
        "target_tokens": [],
    }


def load_model(checkpoint_path: Path, dataset_dir: Path, device: torch.device) -> tuple[TokenGraphDynamicCausalLMV3, LearnedBpeTokenizer, dict[str, Any]]:
    tokenizer = LearnedBpeTokenizer.from_json(json.loads((dataset_dir / "tokenizer.json").read_text(encoding="utf-8")))
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    model = TokenGraphDynamicCausalLMV3(
        vocab_size=len(tokenizer.vocab),
        node_type_count=max(int(v) for v in manifest["node_type_vocab"].values()) + 1,
        edge_type_count=max(int(v) for v in manifest["edge_type_vocab"].values()) + 1,
        dim=int(train_args.get("dim", 512)),
        graph_layers=int(train_args.get("graph_layers", 8)),
        decoder_layers=int(train_args.get("decoder_layers", 10)),
        heads=int(train_args.get("heads", 8)),
        max_answer_tokens=int(train_args.get("max_answer_tokens", 160)),
        dropout=0.0,
        tie_embeddings=bool(train_args.get("tie_embeddings", False) and not train_args.get("untie_embeddings", True)),
        graph_prior_init=float(train_args.get("graph_prior_init", 0.0)),
        graph_prior_max=float(train_args.get("graph_prior_max", 1.0)),
        prefix_window=int(train_args.get("prefix_window", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, tokenizer, train_args


def build_sentence_graph(row: dict[str, Any], tokenizer: LearnedBpeTokenizer, max_target_tokens: int) -> dict[str, Any]:
    return build_graph(
        row,
        tokenizer,
        max_query_tokens=48,
        max_context_tokens=48,
        max_unit_tokens=48,
        max_knowledge_tokens=0,
        max_knowledge_items=0,
        max_target_tokens=max_target_tokens,
        max_segments=1,
        max_same_piece_edges=128,
        max_overlap_edges=64,
        max_semantic_span_nodes=0,
        max_semantic_edge_nodes=0,
        graph_mode="base",
        max_target_prefix_tokens=0,
        include_answer_overlap_hints=False,
    )


@torch.no_grad()
def sentence_loss(model: TokenGraphDynamicCausalLMV3, batch: dict[str, Any], pad_id: int, device: torch.device) -> float:
    moved = move_batch(batch, device)
    outputs = model(moved)
    logits = outputs["logits"]
    target = moved["answer_ids"][:, 1:]
    token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_id, reduction="none")
    mask = target.reshape(-1).ne(pad_id)
    return float(token_loss[mask].mean().detach().cpu()) if bool(mask.any()) else 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-target-tokens", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model(args.checkpoint, args.dataset_dir, device)
    pad_id = int(tokenizer.vocab[PAD])
    df = pd.read_parquet(args.parquet).head(args.limit)
    records: list[dict[str, Any]] = []
    correct = 0
    for idx, row in df.iterrows():
        good = str(row["sentence_good"])
        bad = str(row["sentence_bad"])
        good_graph = build_sentence_graph(make_sentence_row(f"blimp_{idx:05d}_good", good), tokenizer, args.max_target_tokens)
        bad_graph = build_sentence_graph(make_sentence_row(f"blimp_{idx:05d}_bad", bad), tokenizer, args.max_target_tokens)
        loader = DataLoader(PairGraphDataset([good_graph, bad_graph]), batch_size=2, shuffle=False, collate_fn=lambda rows: collate(rows, pad_id=pad_id))
        batch = next(iter(loader))
        moved = move_batch(batch, device)
        outputs = model(moved)
        logits = outputs["logits"]
        target = moved["answer_ids"][:, 1:]
        flat_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_id, reduction="none").reshape(target.shape)
        mask = target.ne(pad_id)
        losses = (flat_loss * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
        good_loss = float(losses[0].detach().cpu())
        bad_loss = float(losses[1].detach().cpu())
        is_correct = good_loss < bad_loss
        correct += int(is_correct)
        records.append(
            {
                "index": int(idx),
                "good": good,
                "bad": bad,
                "good_loss": round(good_loss, 6),
                "bad_loss": round(bad_loss, 6),
                "correct": bool(is_correct),
            }
        )
    payload = {
        "status": "completed",
        "benchmark": "BLiMP likelihood smoke",
        "parquet": str(args.parquet),
        "limit": int(len(records)),
        "accuracy": round(correct / max(1, len(records)), 4),
        "correct": correct,
        "records": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["status", "benchmark", "limit", "accuracy", "correct", "out_json"] if k in payload} | {"out_json": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
