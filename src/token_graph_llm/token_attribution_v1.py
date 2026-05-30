from __future__ import annotations

import argparse
import html
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from generalization_eval_probe_v1 import load_model, make_prompt_sample
from native_token_graph_common import BOS, EOS, PAD, UNK
from train_token_graph_llm_v1 import collate, move_batch


def token_piece(tokenizer: Any, token_id: int) -> str:
    return str(getattr(tokenizer, "id_to_token", {}).get(int(token_id), f"<id:{token_id}>"))


def decode_one(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def sanitize(value: str) -> str:
    return html.escape(str(value), quote=True)


def build_edge_index(sample: dict[str, Any], edge_type_vocab: dict[str, int]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    id_to_edge_type = {int(value): key for key, value in edge_type_vocab.items()}
    edges: list[dict[str, Any]] = []
    incident: dict[int, list[dict[str, Any]]] = {}
    for edge_index, edge in enumerate(sample.get("edges", [])):
        src = int(edge.get("src", 0))
        dst = int(edge.get("dst", 0))
        typed = {
            "index": edge_index,
            "src": src,
            "dst": dst,
            "edge_type_id": int(edge.get("edge_type_id", 0)),
            "edge_type": id_to_edge_type.get(int(edge.get("edge_type_id", 0)), str(edge.get("edge_type_id", 0))),
        }
        edges.append(typed)
        incident.setdefault(src, []).append({**typed, "direction": "out"})
        incident.setdefault(dst, []).append({**typed, "direction": "in"})
    return edges, incident


def nucleus_sample(logits: torch.Tensor, *, temperature: float, top_k: int) -> int:
    if temperature and temperature != 1.0:
        logits = logits / max(float(temperature), 1e-5)
    if top_k and top_k > 0:
        k = min(int(top_k), logits.numel())
        threshold = torch.topk(logits, k=k).values[-1]
        logits = logits.masked_fill(logits < threshold, -1e9)
    return int(torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item())


@torch.no_grad()
def generate_with_attribution(
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    manifest: dict[str, Any],
    *,
    device: torch.device,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    top_nodes: int,
) -> dict[str, Any]:
    pad_id = int(tokenizer.vocab[PAD])
    bos_id = int(tokenizer.vocab[BOS])
    eos_id = int(tokenizer.vocab[EOS])
    unk_id = int(tokenizer.vocab.get(UNK, 3))
    batch = move_batch(collate([sample], pad_id=pad_id), device)
    graph_nodes = model.encode_graph(batch)
    node_mask = batch["node_mask"][0].detach().cpu().tolist()
    node_ids = batch["node_token_ids"][0].detach().cpu().tolist()
    node_types = batch["node_types"][0].detach().cpu().tolist()
    id_to_node_type = {int(value): key for key, value in manifest["node_type_vocab"].items()}
    _, incident = build_edge_index(sample, manifest["edge_type_vocab"])
    nodes = [
        {
            "index": index,
            "piece": token_piece(tokenizer, token_id),
            "text": decode_one(tokenizer, token_id),
            "token_id": int(token_id),
            "node_type_id": int(node_types[index]) if index < len(node_types) else 0,
            "node_type": id_to_node_type.get(int(node_types[index]) if index < len(node_types) else 0, "unknown"),
        }
        for index, token_id in enumerate(node_ids)
        if index < len(node_mask) and bool(node_mask[index])
    ]
    ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    steps: list[dict[str, Any]] = []
    for step in range(max_new_tokens):
        states, path_logits = model.decode_states(batch, graph_nodes, ids)
        logits = model.lm_head(model.lm_norm(states[:, -1]))[0]
        logits[pad_id] = -1e9
        logits[bos_id] = -1e9
        logits[unk_id] = -1e9
        if step < min_new_tokens:
            logits[eos_id] = -1e9
        if repetition_penalty and repetition_penalty > 1.0:
            for token_id in set(int(x) for x in ids[0].tolist()):
                if token_id in {pad_id, bos_id}:
                    continue
                value = logits[token_id]
                logits[token_id] = value / repetition_penalty if value > 0 else value * repetition_penalty
        next_id = nucleus_sample(logits, temperature=temperature, top_k=top_k)
        current_path = path_logits[0, -1]
        weights = torch.softmax(current_path, dim=-1)
        top = torch.topk(weights, k=min(top_nodes, int(batch["node_mask"][0].sum().item())))
        top_node_rows: list[dict[str, Any]] = []
        top_node_set: set[int] = set()
        for rank, (score, node_index) in enumerate(zip(top.values.detach().cpu().tolist(), top.indices.detach().cpu().tolist()), 1):
            node_index = int(node_index)
            top_node_set.add(node_index)
            related_edges = incident.get(node_index, [])[:8]
            top_node_rows.append(
                {
                    "rank": rank,
                    "score": round(float(score), 6),
                    "node": nodes[node_index] if node_index < len(nodes) else {"index": node_index},
                    "incident_edges": related_edges,
                }
            )
        edge_rows: list[dict[str, Any]] = []
        for node_index in sorted(top_node_set):
            for edge in incident.get(node_index, []):
                src = int(edge["src"])
                dst = int(edge["dst"])
                if src in top_node_set or dst in top_node_set:
                    edge_rows.append(edge)
        edge_rows = edge_rows[:16]
        steps.append(
            {
                "step": step + 1,
                "token_id": int(next_id),
                "piece": token_piece(tokenizer, next_id),
                "text": decode_one(tokenizer, next_id),
                "top_nodes": top_node_rows,
                "top_edges": edge_rows,
            }
        )
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == eos_id and step + 1 >= min_new_tokens:
            break
    output_ids = [int(x) for x in ids[0, 1:].detach().cpu().tolist()]
    return {
        "prompt": sample.get("prompt", ""),
        "output": tokenizer.decode(output_ids),
        "output_ids": output_ids,
        "nodes": nodes,
        "edges": build_edge_index(sample, manifest["edge_type_vocab"])[0],
        "steps": steps,
    }


@torch.no_grad()
def force_output_attribution(
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    manifest: dict[str, Any],
    *,
    forced_text: str,
    device: torch.device,
    top_nodes: int,
) -> dict[str, Any]:
    pad_id = int(tokenizer.vocab[PAD])
    bos_id = int(tokenizer.vocab[BOS])
    eos_id = int(tokenizer.vocab[EOS])
    unk_id = int(tokenizer.vocab.get(UNK, 3))
    batch = move_batch(collate([sample], pad_id=pad_id), device)
    graph_nodes = model.encode_graph(batch)
    node_mask = batch["node_mask"][0].detach().cpu().tolist()
    node_ids = batch["node_token_ids"][0].detach().cpu().tolist()
    node_types = batch["node_types"][0].detach().cpu().tolist()
    id_to_node_type = {int(value): key for key, value in manifest["node_type_vocab"].items()}
    _, incident = build_edge_index(sample, manifest["edge_type_vocab"])
    nodes = [
        {
            "index": index,
            "piece": token_piece(tokenizer, token_id),
            "text": decode_one(tokenizer, token_id),
            "token_id": int(token_id),
            "node_type_id": int(node_types[index]) if index < len(node_types) else 0,
            "node_type": id_to_node_type.get(int(node_types[index]) if index < len(node_types) else 0, "unknown"),
        }
        for index, token_id in enumerate(node_ids)
        if index < len(node_mask) and bool(node_mask[index])
    ]
    forced_ids = [token_id for token_id in tokenizer.encode(forced_text) if token_id not in {pad_id, bos_id, eos_id, unk_id}]
    ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    steps: list[dict[str, Any]] = []
    for step, forced_id in enumerate(forced_ids, 1):
        states, path_logits = model.decode_states(batch, graph_nodes, ids)
        logits = model.lm_head(model.lm_norm(states[:, -1]))[0]
        logits[pad_id] = -1e9
        logits[bos_id] = -1e9
        logits[unk_id] = -1e9
        probs = torch.softmax(logits, dim=-1)
        forced_prob = float(probs[int(forced_id)].detach().cpu().item())
        forced_logit = float(logits[int(forced_id)].detach().cpu().item())
        rank = int((logits > logits[int(forced_id)]).sum().detach().cpu().item()) + 1
        current_path = path_logits[0, -1]
        weights = torch.softmax(current_path, dim=-1)
        top = torch.topk(weights, k=min(top_nodes, int(batch["node_mask"][0].sum().item())))
        top_node_rows: list[dict[str, Any]] = []
        top_node_set: set[int] = set()
        for top_rank, (score, node_index) in enumerate(zip(top.values.detach().cpu().tolist(), top.indices.detach().cpu().tolist()), 1):
            node_index = int(node_index)
            top_node_set.add(node_index)
            related_edges = incident.get(node_index, [])[:8]
            top_node_rows.append(
                {
                    "rank": top_rank,
                    "score": round(float(score), 6),
                    "node": nodes[node_index] if node_index < len(nodes) else {"index": node_index},
                    "incident_edges": related_edges,
                }
            )
        edge_rows: list[dict[str, Any]] = []
        for node_index in sorted(top_node_set):
            for edge in incident.get(node_index, []):
                src = int(edge["src"])
                dst = int(edge["dst"])
                if src in top_node_set or dst in top_node_set:
                    edge_rows.append(edge)
        steps.append(
            {
                "step": step,
                "token_id": int(forced_id),
                "piece": token_piece(tokenizer, forced_id),
                "text": decode_one(tokenizer, forced_id),
                "forced_logit": round(forced_logit, 6),
                "forced_prob": round(forced_prob, 8),
                "forced_rank": rank,
                "top_nodes": top_node_rows,
                "top_edges": edge_rows[:16],
            }
        )
        ids = torch.cat([ids, torch.tensor([[int(forced_id)]], dtype=torch.long, device=device)], dim=1)
    return {
        "prompt": sample.get("prompt", ""),
        "forced_text": forced_text,
        "output": tokenizer.decode(forced_ids),
        "output_ids": forced_ids,
        "nodes": nodes,
        "edges": build_edge_index(sample, manifest["edge_type_vocab"])[0],
        "steps": steps,
    }


def render_html(payload: dict[str, Any]) -> str:
    sections: list[str] = []
    for item in payload["items"]:
        node_html = []
        for node in item["nodes"]:
            node_html.append(
                f"<span class='node n{node['index']}' title='#{node['index']} {sanitize(node['node_type'])}'>{node['index']}: {sanitize(node['piece'])}</span>"
            )
        step_html = []
        for step in item["steps"]:
            top_rows = []
            for top in step["top_nodes"]:
                node = top["node"]
                edges = ", ".join(
                    f"{edge['direction']}:{edge['src']}->{edge['dst']}:{sanitize(edge['edge_type'])}" for edge in top.get("incident_edges", [])[:4]
                )
                top_rows.append(
                    f"<tr><td>{top['rank']}</td><td>{top['score']:.4f}</td><td>#{node.get('index')} {sanitize(node.get('piece', ''))}</td><td>{sanitize(node.get('text', ''))}</td><td>{edges}</td></tr>"
                )
            step_html.append(
                f"""
                <article class='step'>
                  <h3>Step {step['step']} -> <code>{sanitize(step['piece'])}</code> <span>{sanitize(step['text'])}</span></h3>
                  <p class='meta'>forced rank: {sanitize(step.get('forced_rank', 'sampled'))}; forced prob: {sanitize(step.get('forced_prob', 'sampled'))}</p>
                  <table>
                    <thead><tr><th>rank</th><th>path weight</th><th>graph node</th><th>decoded text</th><th>incident edges</th></tr></thead>
                    <tbody>{''.join(top_rows)}</tbody>
                  </table>
                </article>
                """
            )
        sections.append(
            f"""
            <section class='case'>
              <h2>{sanitize(item['label'])}</h2>
              <p><b>Prompt:</b> {sanitize(item['prompt'])}</p>
              <p><b>Output:</b> {sanitize(item['output'])}</p>
              <div class='nodes'>{''.join(node_html)}</div>
              {''.join(step_html)}
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Token Graph LLM Attribution</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; background: #f7f7f4; color: #171717; }}
header {{ padding: 28px 36px; background: #111; color: #fff; }}
main {{ padding: 24px 36px 48px; }}
.case {{ margin: 0 0 28px; padding: 20px; background: #fff; border: 1px solid #ddd; border-radius: 8px; }}
.nodes {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0 18px; padding: 12px; background: #fafafa; border: 1px solid #e6e6e6; }}
.node {{ display: inline-block; padding: 5px 8px; border-radius: 5px; background: #e8eef7; font-size: 12px; }}
.step {{ margin: 14px 0; padding: 14px; border-left: 4px solid #222; background: #fbfbfb; }}
.meta {{ color: #555; font-size: 13px; }}
h1, h2, h3 {{ margin: 0 0 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; vertical-align: top; border-bottom: 1px solid #e5e5e5; padding: 7px 8px; }}
code {{ background: #eee; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<header>
  <h1>Token Attribution 可视化</h1>
  <p>每个生成 token 对应的 top graph nodes / incident edges。权重来自模型内部 path logits，不是外部 LLM 判断。</p>
</header>
<main>
{''.join(sections)}
</main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-html", required=True, type=Path)
    parser.add_argument("--max-nodes", type=int, default=384)
    parser.add_argument("--max-edges", type=int, default=1536)
    parser.add_argument("--max-sequence-tokens", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--min-new-tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--top-nodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--forced-errors", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, manifest, _ = load_model(args.run_dir, args.dataset_dir, device)
    prompts = [
        ("color_question_orang", "A green lantern is on the wooden desk. What color is the lantern?", "Orang"),
        ("pencil_count_minnesot", "Mira packed seven pencils and two erasers. How many pencils did Mira pack?", "Minnesot"),
    ]
    items: list[dict[str, Any]] = []
    for label, prompt, forced_text in prompts:
        sample = make_prompt_sample(
            prompt,
            tokenizer,
            manifest,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_sequence_tokens=args.max_sequence_tokens,
        )
        if args.forced_errors:
            item = force_output_attribution(
                model,
                tokenizer,
                sample,
                manifest,
                forced_text=forced_text,
                device=device,
                top_nodes=args.top_nodes,
            )
        else:
            item = generate_with_attribution(
                model,
                tokenizer,
                sample,
                manifest,
                device=device,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                top_nodes=args.top_nodes,
            )
        item["label"] = label
        items.append(item)
    payload = {"status": "completed", "device": str(device), "run_dir": str(args.run_dir), "items": items}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"status": "completed", "out_json": str(args.out_json), "out_html": str(args.out_html), "items": [{"label": item["label"], "output": item["output"]} for item in items]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
