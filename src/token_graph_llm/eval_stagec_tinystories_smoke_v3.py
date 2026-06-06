from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from build_native_token_reasoning_graph_dataset_v3 import build_graph
from eval_dynamic_v3_compare_ablation import make_variant
from model_token_graph_dynamic_decoder_v3 import TokenGraphDynamicCausalLMV3
from native_token_graph_common import BOS, EOS, PAD, UNK, LearnedBpeTokenizer
from train_graph_causal_decoder_v2 import collate, move_batch


def split_tinystories_text(text: str) -> list[str]:
    marker = "<|endoftext|>"
    if marker in text:
        return [chunk.strip() for chunk in text.split(marker) if chunk.strip()]
    chunks = [chunk.strip() for chunk in text.split("\n\n\n") if chunk.strip()]
    return chunks if len(chunks) > 1 else [text.strip()]


def load_tinystories_file(path: Path, count: int, prompt_tokens: int) -> list[dict[str, str]]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= count * 3:
                    break
    else:
        stories = split_tinystories_text(path.read_text(encoding="utf-8", errors="ignore"))
        rows = [{"id": f"tinystories_file_{idx:05d}", "text": story} for idx, story in enumerate(stories)]
    return select_story_prefixes(rows, count, prompt_tokens)


def load_tinystories_hf(count: int, prompt_tokens: int) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install `datasets`, or pass --tinystories-file to read TinyStories-valid.txt/jsonl directly") from exc
    rows = load_dataset("roneneldan/TinyStories", split=f"validation[:{count}]")
    return select_story_prefixes(rows, count, prompt_tokens)


def select_story_prefixes(rows: Any, count: int, prompt_tokens: int) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        words = text.split()
        if len(words) < prompt_tokens + 12:
            continue
        prompt = " ".join(words[:prompt_tokens])
        gold = " ".join(words[prompt_tokens : prompt_tokens + 80])
        sample_id = str(row.get("id", f"tinystories_{idx:04d}"))
        samples.append({"sample_id": sample_id, "prompt": prompt, "gold_prefix": gold})
        if len(samples) >= count:
            break
    return samples[:count]


def make_schema_row(sample: dict[str, str]) -> dict[str, Any]:
    prompt = sample["prompt"]
    return {
        "schema_version": "token_graph_corpus_v2",
        "sample_id": sample["sample_id"],
        "source": "tinystories_validation_smoke",
        "split": "probe",
        "query": "Continue the story in natural language.",
        "question_date": None,
        "source_segments": [
            {
                "segment_id": "story_prefix",
                "text": prompt,
                "segment_type": "story_prefix",
                "timestamp": None,
                "source_port": "tinystories_validation",
            }
        ],
        "text_units": [
            {
                "unit_id": "story_prefix:u1",
                "parent_segment_id": "story_prefix",
                "text": prompt,
                "unit_type": "story_prefix",
                "numeric_surface": "",
                "temporal_surface": "",
                "state_surface": "",
                "source_segment_type": "story_prefix",
            }
        ],
        "support_segment_ids": ["story_prefix"],
        "support_text_unit_ids": ["story_prefix:u1"],
        "support_alignment": [],
        "target_text": "",
        "target_tokens": [],
    }


class GraphRows:
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
            "answer": row.get("gold_prefix", ""),
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
            "answer_ids": [],
        }


def load_model(checkpoint_path: Path, dataset_dir: Path, device: torch.device) -> tuple[TokenGraphDynamicCausalLMV3, LearnedBpeTokenizer]:
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
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        print(json.dumps({"event": "load_warning", "missing": missing, "unexpected": unexpected}), flush=True)
    model.eval()
    return model, tokenizer


def text_stats(text: str) -> dict[str, Any]:
    words = text.split()
    unique = len(set(words))
    repeated_bigrams = 0
    bigrams = list(zip(words, words[1:]))
    if bigrams:
        repeated_bigrams = len(bigrams) - len(set(bigrams))
    return {
        "chars": len(text),
        "words": len(words),
        "unique_word_ratio": round(unique / max(1, len(words)), 4),
        "repeated_bigrams": repeated_bigrams,
    }


@torch.no_grad()
def generate_one(
    model: TokenGraphDynamicCausalLMV3,
    tokenizer: LearnedBpeTokenizer,
    batch: dict[str, Any],
    *,
    variant: str,
    pad_id: int,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    moved = move_batch(make_variant(batch, variant, pad_id), device)
    ids = model.generate(
        moved,
        bos_id=int(tokenizer.vocab[BOS]),
        eos_id=int(tokenizer.vocab[EOS]),
        pad_id=pad_id,
        unk_id=int(tokenizer.vocab.get(UNK, 3)),
        max_new_tokens=max_new_tokens,
        min_new_tokens=12,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    )[0].detach().cpu().tolist()
    return tokenizer.decode(ids)


def lexical_overlap(a: str, b: str) -> float:
    aw = {w.strip(".,!?;:\"'").lower() for w in a.split() if len(w.strip(".,!?;:\"'")) > 2}
    bw = {w.strip(".,!?;:\"'").lower() for w in b.split() if len(w.strip(".,!?;:\"'")) > 2}
    return round(len(aw & bw) / max(1, len(aw | bw)), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--tinystories-file", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=45)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-query-tokens", type=int, default=96)
    parser.add_argument("--max-context-tokens", type=int, default=192)
    parser.add_argument("--max-unit-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, args.dataset_dir, device)
    pad_id = int(tokenizer.vocab[PAD])
    samples = (
        load_tinystories_file(args.tinystories_file, args.sample_count, args.prompt_tokens)
        if args.tinystories_file
        else load_tinystories_hf(args.sample_count, args.prompt_tokens)
    )
    graph_rows = []
    for sample in samples:
        graph = build_graph(
            make_schema_row(sample),
            tokenizer,
            max_query_tokens=args.max_query_tokens,
            max_context_tokens=args.max_context_tokens,
            max_unit_tokens=args.max_unit_tokens,
            max_knowledge_tokens=0,
            max_knowledge_items=0,
            max_target_tokens=1,
            max_segments=2,
            max_same_piece_edges=256,
            max_overlap_edges=128,
            max_semantic_span_nodes=0,
            max_semantic_edge_nodes=0,
            include_answer_overlap_hints=False,
        )
        graph["gold_prefix"] = sample["gold_prefix"]
        graph["story_prompt"] = sample["prompt"]
        graph_rows.append(graph)

    loader = DataLoader(GraphRows(graph_rows), batch_size=1, shuffle=False, collate_fn=lambda rows: collate(rows, pad_id=pad_id))
    variants = ["normal", "no_edges", "shuffle_edges"]
    results: list[dict[str, Any]] = []
    for idx, (batch, graph) in enumerate(zip(loader, graph_rows)):
        item: dict[str, Any] = {
            "sample_id": graph["sample_id"],
            "prompt": graph["story_prompt"],
            "gold_prefix": graph["gold_prefix"],
            "variants": {},
        }
        normal_text = ""
        for variant in variants:
            text = generate_one(
                model,
                tokenizer,
                batch,
                variant=variant,
                pad_id=pad_id,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed + idx,
            )
            if variant == "normal":
                normal_text = text
            item["variants"][variant] = {
                "text": text,
                "stats": text_stats(text),
                "gold_overlap": lexical_overlap(text, graph["gold_prefix"]),
            }
        for variant in variants:
            item["variants"][variant]["normal_overlap"] = lexical_overlap(item["variants"][variant]["text"], normal_text)
        results.append(item)

    aggregate: dict[str, Any] = {"sample_count": len(results), "variants": {}}
    for variant in variants:
        rows = [item["variants"][variant] for item in results]
        aggregate["variants"][variant] = {
            "avg_words": round(sum(row["stats"]["words"] for row in rows) / max(1, len(rows)), 2),
            "avg_unique_word_ratio": round(sum(row["stats"]["unique_word_ratio"] for row in rows) / max(1, len(rows)), 4),
            "avg_repeated_bigrams": round(sum(row["stats"]["repeated_bigrams"] for row in rows) / max(1, len(rows)), 2),
            "avg_gold_overlap": round(sum(row["gold_overlap"] for row in rows) / max(1, len(rows)), 4),
            "avg_normal_overlap": round(sum(row["normal_overlap"] for row in rows) / max(1, len(rows)), 4),
        }

    payload = {
        "status": "completed",
        "benchmark": "TinyStories validation smoke",
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "aggregate": aggregate,
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "out_json": str(args.out_json), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
