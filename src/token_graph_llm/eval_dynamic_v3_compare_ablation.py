from __future__ import annotations

import argparse
import copy
import html
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from model_token_graph_dynamic_decoder_v3 import TokenGraphDynamicCausalLMV3, graph_lm_loss_v3
from native_token_graph_common import BOS, EOS, PAD, UNK, LearnedBpeTokenizer
from train_graph_causal_decoder_v2 import NativeTokenGraphDataset, collate, move_batch, resolve_split_paths


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in batch.items()}


def make_variant(batch: dict[str, Any], variant: str, pad_id: int) -> dict[str, Any]:
    out = clone_batch(batch)
    if variant == "normal":
        return out
    if variant == "no_edges":
        out["edge_mask"].zero_()
        return out
    if variant == "shuffle_edges":
        for row in range(out["edge_src"].size(0)):
            valid = out["edge_mask"][row].nonzero(as_tuple=False).flatten()
            if valid.numel() > 1:
                perm = valid[torch.randperm(valid.numel(), device=valid.device)]
                out["edge_src"][row, valid] = out["edge_src"][row, perm]
                out["edge_dst"][row, valid] = out["edge_dst"][row, perm.roll(1)]
        return out
    if variant == "mask_context":
        out["node_mask"].zero_()
        out["edge_mask"].zero_()
        out["node_token_ids"].fill_(pad_id)
        return out
    raise ValueError(f"unknown variant: {variant}")


def load_v3_model(checkpoint_path: Path, dataset_dir: Path, device: torch.device) -> tuple[TokenGraphDynamicCausalLMV3, dict[str, Any]]:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    model = TokenGraphDynamicCausalLMV3(
        vocab_size=int(args.get("vocab_size", 50261)) if "vocab_size" in args else len(
            LearnedBpeTokenizer.from_json(json.loads((dataset_dir / "tokenizer.json").read_text(encoding="utf-8"))).vocab
        ),
        node_type_count=max(int(v) for v in manifest["node_type_vocab"].values()) + 1,
        edge_type_count=max(int(v) for v in manifest["edge_type_vocab"].values()) + 1,
        dim=int(args.get("dim", 512)),
        graph_layers=int(args.get("graph_layers", 8)),
        decoder_layers=int(args.get("decoder_layers", 10)),
        heads=int(args.get("heads", 8)),
        max_answer_tokens=int(args.get("max_answer_tokens", 160)),
        dropout=0.0,
        tie_embeddings=bool(args.get("tie_embeddings", False) and not args.get("untie_embeddings", True)),
        graph_prior_init=float(args.get("graph_prior_init", 0.0)),
        graph_prior_max=float(args.get("graph_prior_max", 1.0)),
        prefix_window=int(args.get("prefix_window", 64)),
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        print(json.dumps({"event": "load_warning", "checkpoint": str(checkpoint_path), "missing": missing, "unexpected": unexpected}), flush=True)
    model.eval()
    return model, args


def average_losses(
    model: TokenGraphDynamicCausalLMV3,
    loader: DataLoader,
    *,
    variant: str,
    pad_id: int,
    device: torch.device,
    max_batches: int,
    label_smoothing: float,
    graph_state_weight: float,
    next_token_node_weight: float,
    edge_type_weight: float,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            moved = move_batch(make_variant(batch, variant, pad_id), device)
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" and amp_dtype is not None else torch.no_grad()
            with ctx:
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


def decode_piece(tokenizer: LearnedBpeTokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)])
    return text if text else f"<id:{token_id}>"


@torch.no_grad()
def generate_text(
    model: TokenGraphDynamicCausalLMV3,
    batch: dict[str, Any],
    tokenizer: LearnedBpeTokenizer,
    *,
    variant: str,
    pad_id: int,
    device: torch.device,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
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
        min_new_tokens=3,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
    )[0].detach().cpu().tolist()
    return tokenizer.decode(ids)


@torch.no_grad()
def attribution_trace(
    model: TokenGraphDynamicCausalLMV3,
    batch: dict[str, Any],
    tokenizer: LearnedBpeTokenizer,
    *,
    pad_id: int,
    device: torch.device,
    max_new_tokens: int,
    top_nodes: int,
    top_tokens: int,
) -> dict[str, Any]:
    moved = move_batch(make_variant(batch, "normal", pad_id), device)
    encoded = model.encode_context(moved)
    context = encoded["node_states"]
    context_mask = moved["node_mask"]
    context_prior = (model.context_token_score(context).squeeze(-1) + model.answer_overlap_score(context).squeeze(-1)).masked_fill(~context_mask, -1e4)
    ids = torch.full((context.size(0), 1), int(tokenizer.vocab[BOS]), dtype=torch.long, device=device)
    trace: list[dict[str, Any]] = []
    node_ids = moved["node_token_ids"][0]
    for step in range(max_new_tokens):
        generated, decoder_tunnel_logits = model._decode_answer_nodes(ids, context, context_mask, context_prior)
        current = generated[:, -1:]
        tunnel_logits = torch.einsum("btd,bnd->btn", model.tunnel_query(current), model.tunnel_key(context)) / math.sqrt(context.size(-1))
        tunnel_logits = tunnel_logits + decoder_tunnel_logits[:, -1:, :]
        tunnel_logits = tunnel_logits.masked_fill(~context_mask.unsqueeze(1), -1e4)
        node_weights = torch.softmax((tunnel_logits.squeeze(1) + context_prior)[0], dim=-1)
        logits = model.lm_head(model.lm_norm(generated[:, -1]))
        logits[:, pad_id] = -1e9
        logits[:, int(tokenizer.vocab[BOS])] = -1e9
        logits[:, int(tokenizer.vocab.get(UNK, 3))] = -1e9
        if step < 3:
            logits[:, int(tokenizer.vocab[EOS])] = -1e9
        token_probs = torch.softmax(logits[0], dim=-1)
        chosen = int(torch.argmax(logits[0]).detach().cpu())
        tok_vals, tok_idxs = torch.topk(token_probs, k=min(top_tokens, token_probs.numel()))
        node_vals, node_idxs = torch.topk(node_weights, k=min(top_nodes, int(context_mask[0].sum().detach().cpu()) or 1))
        trace.append(
            {
                "step": step + 1,
                "chosen_id": chosen,
                "chosen_piece": decode_piece(tokenizer, chosen),
                "top_next_tokens": [
                    {"piece_id": int(idx.detach().cpu()), "piece": decode_piece(tokenizer, int(idx.detach().cpu())), "prob": round(float(val.detach().cpu()), 6)}
                    for val, idx in zip(tok_vals, tok_idxs)
                ],
                "top_graph_nodes": [
                    {
                        "node_index": int(idx.detach().cpu()),
                        "piece_id": int(node_ids[int(idx.detach().cpu())].detach().cpu()),
                        "piece": decode_piece(tokenizer, int(node_ids[int(idx.detach().cpu())].detach().cpu())),
                        "weight": round(float(val.detach().cpu()), 6),
                    }
                    for val, idx in zip(node_vals, node_idxs)
                ],
            }
        )
        ids = torch.cat([ids, torch.tensor([[chosen]], dtype=torch.long, device=device)], dim=1)
        if chosen == int(tokenizer.vocab[EOS]) and step >= 3:
            break
    return {"generated_text": tokenizer.decode(ids[0, 1:].detach().cpu().tolist()), "trace": trace}


def render_html(payload: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    cards = []
    for sample in payload.get("attribution", []):
        rows = []
        for step in sample["trace"]:
            toks = "".join(f"<span class='pill'>{esc(t['piece'])}<small>{t['prob']}</small></span>" for t in step["top_next_tokens"])
            nodes = "".join(f"<div class='node'>#{n['node_index']} {esc(n['piece'])}<small>{n['weight']}</small></div>" for n in step["top_graph_nodes"])
            rows.append(f"<tr><td>{step['step']}</td><td class='chosen'>{esc(step['chosen_piece'])}</td><td>{toks}</td><td>{nodes}</td></tr>")
        cards.append(
            f"<section><h2>{esc(sample['query'])}</h2><p><b>Gold:</b> {esc(sample['gold'])}</p>"
            f"<p><b>Generated:</b> {esc(sample['generated_text'])}</p>"
            "<table><thead><tr><th>Step</th><th>Chosen</th><th>Top next tokens</th><th>Top graph nodes</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    return (
        "<!doctype html><meta charset='utf-8'><title>TMCRA TokenGraph-LLM v3 attribution</title>"
        "<style>body{font-family:Arial,sans-serif;background:#f6f7f9;color:#111;margin:0;padding:24px}"
        "section{background:white;border:1px solid #ddd;border-radius:8px;margin:18px 0;padding:16px}"
        "table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{border-top:1px solid #eee;padding:8px;vertical-align:top}"
        ".chosen{font-weight:700;color:#0a57a3}.pill{display:inline-flex;gap:6px;border:1px solid #ccd;padding:3px 7px;border-radius:999px;margin:2px}"
        ".pill small,.node small{color:#667;margin-left:6px}.node{border-left:3px solid #0a57a3;background:#f3f8ff;margin:3px 0;padding:5px}</style>"
        "<h1>TMCRA TokenGraph-LLM v3 Token Attribution</h1>"
        f"<p>Run: {esc(payload.get('attribution_run'))}</p>{''.join(cards)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--checkpoints-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-html", required=True, type=Path)
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--metric-limit", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-nodes", type=int, default=384)
    parser.add_argument("--max-edges", type=int, default=1536)
    parser.add_argument("--max-answer-tokens", type=int, default=160)
    parser.add_argument("--max-indexed-lines-per-shard", type=int, default=20000)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--attribution-label", default="StageC")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LearnedBpeTokenizer.from_json(json.loads((args.dataset_dir / "tokenizer.json").read_text(encoding="utf-8")))
    pad_id = int(tokenizer.vocab[PAD])
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    val_paths = resolve_split_paths(args.dataset_dir, manifest, "val", "val.jsonl", "val.base.jsonl")
    metric_ds = NativeTokenGraphDataset(
        val_paths,
        limit=args.metric_limit,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_answer_tokens=args.max_answer_tokens,
        seed=args.seed,
        max_indexed_lines_per_shard=args.max_indexed_lines_per_shard,
    )
    sample_ds = NativeTokenGraphDataset(
        val_paths,
        limit=args.sample_limit,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_answer_tokens=args.max_answer_tokens,
        seed=args.seed,
        max_indexed_lines_per_shard=args.max_indexed_lines_per_shard,
    )
    metric_loader = DataLoader(metric_ds, batch_size=4, shuffle=False, collate_fn=lambda rows: collate(rows, pad_id=pad_id))
    sample_loader = DataLoader(sample_ds, batch_size=1, shuffle=False, collate_fn=lambda rows: collate(rows, pad_id=pad_id))
    checkpoints = json.loads(args.checkpoints_json.read_text(encoding="utf-8"))
    if not isinstance(checkpoints, dict):
        raise TypeError("checkpoints-json must be an object mapping label to checkpoint path")

    variants = ["normal", "no_edges", "shuffle_edges"]
    payload: dict[str, Any] = {
        "status": "completed",
        "device": str(device),
        "dataset_dir": str(args.dataset_dir),
        "variants": variants,
        "metrics": {},
        "generations": [],
        "attribution_run": args.attribution_label,
        "attribution": [],
    }
    loaded_models: dict[str, TokenGraphDynamicCausalLMV3] = {}
    for label, checkpoint in checkpoints.items():
        model, train_args = load_v3_model(Path(checkpoint), args.dataset_dir, device)
        loaded_models[label] = model
        payload["metrics"][label] = {}
        for variant in variants:
            payload["metrics"][label][variant] = average_losses(
                model,
                metric_loader,
                variant=variant,
                pad_id=pad_id,
                device=device,
                max_batches=args.max_val_batches,
                label_smoothing=float(train_args.get("label_smoothing", 0.02)),
                graph_state_weight=float(train_args.get("graph_state_weight", 0.35)),
                next_token_node_weight=float(train_args.get("next_token_node_weight", 0.08)),
                edge_type_weight=float(train_args.get("edge_type_weight", 0.05)),
                amp_dtype=torch.bfloat16,
            )

    for idx, batch in enumerate(sample_loader):
        row: dict[str, Any] = {"index": idx, "query": batch["queries"][0], "gold": batch["answers"][0], "models": {}}
        for label, model in loaded_models.items():
            row["models"][label] = {
                variant: generate_text(
                    model,
                    batch,
                    tokenizer,
                    variant=variant,
                    pad_id=pad_id,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    seed=args.seed + idx,
                )
                for variant in variants
            }
        payload["generations"].append(row)

    if args.attribution_label in loaded_models:
        attribution_model = loaded_models[args.attribution_label]
        for idx, batch in enumerate(sample_loader):
            if idx >= min(3, args.sample_limit):
                break
            item = {
                "index": idx,
                "query": batch["queries"][0],
                "gold": batch["answers"][0],
                **attribution_trace(
                    attribution_model,
                    batch,
                    tokenizer,
                    pad_id=pad_id,
                    device=device,
                    max_new_tokens=min(24, args.max_new_tokens),
                    top_nodes=6,
                    top_tokens=8,
                ),
            }
            payload["attribution"].append(item)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_html.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"status": "completed", "out_json": str(args.out_json), "out_html": str(args.out_html)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
