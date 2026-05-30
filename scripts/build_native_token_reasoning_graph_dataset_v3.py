from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from native_token_graph_common import BOS, EOS, HuggingFaceBpeTokenizer, LearnedBpeTokenizer, write_jsonl


NODE_TYPES = {
    "query_token": 1,
    "context_token": 2,
    "unit_token": 3,
}

EDGE_TYPES = {
    "next_token": 1,
    "prev_token": 2,
    "same_piece": 3,
    "query_context_overlap": 4,
    "query_unit_overlap": 5,
    "answer_overlap_hint": 6,
}


def collect_texts(rows: list[dict[str, Any]], *, max_text_chars: int) -> list[str]:
    texts: list[str] = []
    for row in rows:
        texts.append(str(row.get("query", "") or "")[:max_text_chars])
        texts.append(_target_text(row)[:max_text_chars])
        for node in _source_segments(row):
            texts.append(str(node.get("text", "") or "")[:max_text_chars])
        for unit in _text_units(row):
            texts.append(str(unit.get("text", "") or unit.get("content", "") or "")[:max_text_chars])
    return [text for text in texts if text.strip()]


def load_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def cap_tokenizer_texts(texts: list[str], *, text_limit: int, char_budget: int) -> list[str]:
    capped = texts[: text_limit] if text_limit and len(texts) > text_limit else list(texts)
    if not char_budget:
        return capped
    out: list[str] = []
    used = 0
    for text in capped:
        remaining = char_budget - used
        if remaining <= 0:
            break
        clipped = text[:remaining]
        if clipped.strip():
            out.append(clipped)
            used += len(clipped)
    return out


def _target_text(row: dict[str, Any]) -> str:
    if row.get("target_text") is not None:
        return str(row.get("target_text", "") or "")
    legacy = row.get("legacy", {}) or {}
    return str(legacy.get("answer", "") or "")


def _source_segments(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("source_segments"), list):
        return list(row.get("source_segments") or [])
    legacy = row.get("legacy", {}) or {}
    raw_nodes = legacy.get("memory_nodes", [])
    segments: list[dict[str, Any]] = []
    for node in raw_nodes or []:
        segments.append(
            {
                "segment_id": str(node.get("node_id", "")),
                "segment_type": str(node.get("speaker", "")),
                "text": str(node.get("text", "") or ""),
                "source_port": str(node.get("source_port", "")),
                "timestamp": node.get("timestamp"),
            }
        )
    return segments


def _text_units(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("text_units"), list):
        return list(row.get("text_units") or [])
    legacy = row.get("legacy", {}) or {}
    raw_units = legacy.get("event_units", [])
    units: list[dict[str, Any]] = []
    for unit in raw_units or []:
        units.append(
            {
                "unit_id": str(unit.get("unit_id", "")),
                "parent_segment_id": str(unit.get("parent_segment_id", "") or unit.get("parent_node_id", "")),
                "text": str(unit.get("text", "") or unit.get("content", "") or ""),
            }
        )
    return units


def _support_segment_ids(row: dict[str, Any]) -> set[str]:
    if row.get("support_segment_ids") is not None:
        return set(str(x) for x in (row.get("support_segment_ids", []) or []))
    legacy = row.get("legacy", {}) or {}
    return set(str(x) for x in (legacy.get("support_ids", []) or []))


def _support_text_unit_ids(row: dict[str, Any]) -> set[str]:
    if row.get("support_text_unit_ids") is not None:
        return set(str(x) for x in (row.get("support_text_unit_ids", []) or []))
    legacy = row.get("legacy", {}) or {}
    return set(str(x) for x in (legacy.get("support_unit_ids", []) or []))


def _add_segment(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    tokenizer: LearnedBpeTokenizer,
    sample_id: str,
    segment_id: str,
    segment_type: str,
    source_id: str,
    text: str,
    max_tokens: int,
) -> list[int]:
    pieces = tokenizer.encode_pieces(text, max_tokens=max_tokens)
    piece_ids = [tokenizer.vocab.get(piece, tokenizer.vocab.get("<unk>", 3)) for piece in pieces]
    indices: list[int] = []
    prev: int | None = None
    for pos, (piece, piece_id) in enumerate(zip(pieces, piece_ids)):
        index = len(nodes)
        indices.append(index)
        nodes.append(
            {
                "id": f"{sample_id}:{segment_id}:tok:{pos:04d}",
                "piece": piece,
                "piece_id": int(piece_id),
                "node_type": segment_type,
                "node_type_id": NODE_TYPES[segment_type],
                "segment_id": segment_id,
                "source_id": source_id,
                "position": pos,
                "text": piece,
                "support_label": 0,
                "answer_overlap_label": 0,
                "teacher_support_label": 0,
                "teacher_negative_label": 0,
            }
        )
        if prev is not None:
            edges.append({"src": prev, "dst": index, "edge_type": "next_token", "edge_type_id": EDGE_TYPES["next_token"], "label": 1})
            edges.append({"src": index, "dst": prev, "edge_type": "prev_token", "edge_type_id": EDGE_TYPES["prev_token"], "label": 1})
        prev = index
    return indices


def _bounded_pair_edges(
    edges: list[dict[str, Any]],
    left: list[int],
    right: list[int],
    *,
    left_piece_ids: list[int],
    right_piece_ids: list[int],
    edge_type: str,
    max_edges: int,
) -> None:
    added = 0
    right_by_piece: dict[int, list[int]] = defaultdict(list)
    for idx, piece_id in zip(right, right_piece_ids):
        right_by_piece[int(piece_id)].append(idx)
    for l_idx, l_piece_id in zip(left, left_piece_ids):
        for r_idx in right_by_piece.get(int(l_piece_id), [])[:4]:
            edges.append({"src": l_idx, "dst": r_idx, "edge_type": edge_type, "edge_type_id": EDGE_TYPES[edge_type], "label": 0})
            edges.append({"src": r_idx, "dst": l_idx, "edge_type": edge_type, "edge_type_id": EDGE_TYPES[edge_type], "label": 0})
            added += 2
            if added >= max_edges:
                return


def _segments_for_annotation(row: dict[str, Any], *, max_segments: int, max_segment_chars: int) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for idx, node in enumerate(_source_segments(row)):
        text = str(node.get("text", "") or "").strip()
        if text:
            segments.append({"segment_id": f"context_{idx:03d}", "kind": "source_segment", "text": text[:max_segment_chars]})
    for idx, unit in enumerate(_text_units(row)):
        text = str(unit.get("text", "") or unit.get("content", "") or "").strip()
        if text:
            segments.append({"segment_id": f"unit_{idx:03d}", "kind": "text_unit", "text": text[:max_segment_chars]})
    return segments[:max_segments]


def build_graph(
    row: dict[str, Any],
    tokenizer: LearnedBpeTokenizer,
    *,
    max_query_tokens: int,
    max_context_tokens: int,
    max_unit_tokens: int,
    max_segments: int,
    max_same_piece_edges: int,
    max_overlap_edges: int,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id", "") or "")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    query_indices = _add_segment(
        nodes=nodes,
        edges=edges,
        tokenizer=tokenizer,
        sample_id=sample_id,
        segment_id="query",
        segment_type="query_token",
        source_id="query",
        text=str(row.get("query", "") or ""),
        max_tokens=max_query_tokens,
    )
    segment_index: dict[str, list[int]] = {"query": query_indices}
    source_supports = _support_segment_ids(row)
    source_supports |= _support_text_unit_ids(row)

    for idx, memory in enumerate(_source_segments(row)[:max_segments]):
        segment_id = f"context_{idx:03d}"
        source_id = str(memory.get("segment_id", segment_id))
        segment_index[segment_id] = _add_segment(
            nodes=nodes,
            edges=edges,
            tokenizer=tokenizer,
            sample_id=sample_id,
            segment_id=segment_id,
            segment_type="context_token",
            source_id=source_id,
            text=str(memory.get("text", "") or ""),
            max_tokens=max_context_tokens,
        )
        if source_id in source_supports:
            for node_idx in segment_index[segment_id]:
                nodes[node_idx]["support_label"] = 1

    for idx, unit in enumerate(_text_units(row)[:max_segments]):
        segment_id = f"unit_{idx:03d}"
        source_id = str(unit.get("unit_id", segment_id))
        segment_index[segment_id] = _add_segment(
            nodes=nodes,
            edges=edges,
            tokenizer=tokenizer,
            sample_id=sample_id,
            segment_id=segment_id,
            segment_type="unit_token",
            source_id=source_id,
            text=str(unit.get("text", "") or unit.get("content", "") or ""),
            max_tokens=max_unit_tokens,
        )
        if source_id in source_supports:
            for node_idx in segment_index[segment_id]:
                nodes[node_idx]["support_label"] = 1

    answer_ids = tokenizer.encode(_target_text(row), max_tokens=256)
    answer_set = set(answer_ids)
    for node in nodes:
        if int(node["piece_id"]) in answer_set:
            node["answer_overlap_label"] = 1

    by_piece: dict[int, list[int]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        by_piece[int(node["piece_id"])].append(idx)
    added = 0
    for indices in by_piece.values():
        if len(indices) < 2:
            continue
        for left, right in zip(indices, indices[1:]):
            edges.append({"src": left, "dst": right, "edge_type": "same_piece", "edge_type_id": EDGE_TYPES["same_piece"], "label": 0})
            edges.append({"src": right, "dst": left, "edge_type": "same_piece", "edge_type_id": EDGE_TYPES["same_piece"], "label": 0})
            added += 2
            if added >= max_same_piece_edges:
                break
        if added >= max_same_piece_edges:
            break

    query_piece_ids = [int(nodes[i]["piece_id"]) for i in query_indices]
    for segment_id, indices in segment_index.items():
        if segment_id == "query":
            continue
        piece_ids = [int(nodes[i]["piece_id"]) for i in indices]
        edge_type = "query_unit_overlap" if segment_id.startswith("unit_") else "query_context_overlap"
        _bounded_pair_edges(
            edges,
            query_indices,
            indices,
            left_piece_ids=query_piece_ids,
            right_piece_ids=piece_ids,
            edge_type=edge_type,
            max_edges=max_overlap_edges,
        )
    for idx, node in enumerate(nodes):
        if int(node["piece_id"]) in answer_set:
            for q_idx in query_indices[:8]:
                edges.append({"src": q_idx, "dst": idx, "edge_type": "answer_overlap_hint", "edge_type_id": EDGE_TYPES["answer_overlap_hint"], "label": 0})

    answer_target_ids = [tokenizer.vocab[BOS]] + answer_ids + [tokenizer.vocab[EOS]]
    return {
        "dataset_version": "native_token_reasoning_graph_v3_base",
        "sample_id": sample_id,
        "source": str(row.get("source", "") or ""),
        "query": str(row.get("query", "") or ""),
        "answer": _target_text(row),
        "target_text": _target_text(row),
        "target_ids": answer_target_ids,
        "answer_ids": answer_target_ids,
        "nodes": nodes,
        "edges": edges,
        "segment_index": segment_index,
        "node_type_vocab": NODE_TYPES,
        "edge_type_vocab": EDGE_TYPES,
        "annotation": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tokenizer-kind", choices=["hf_bpe", "char_bpe"], default="hf_bpe")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-pair-freq", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=512)
    parser.add_argument("--tokenizer-text-limit", type=int, default=1000)
    parser.add_argument("--tokenizer-char-budget", type=int, default=250000)
    parser.add_argument("--max-query-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=128)
    parser.add_argument("--max-unit-tokens", type=int, default=128)
    parser.add_argument("--max-segments", type=int, default=32)
    parser.add_argument("--max-same-piece-edges", type=int, default=384)
    parser.add_argument("--max-overlap-edges", type=int, default=192)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    started = time.perf_counter()
    print(f"[stage] loading rows from {args.input_jsonl}", flush=True)
    rows = load_rows(args.input_jsonl, limit=args.limit)
    print(f"[stage] loaded rows={len(rows)} elapsed={time.perf_counter() - started:.2f}s", flush=True)
    train_texts = collect_texts(rows, max_text_chars=args.max_text_chars)
    raw_train_text_count = len(train_texts)
    raw_train_char_count = sum(len(text) for text in train_texts)
    train_texts = cap_tokenizer_texts(
        train_texts,
        text_limit=args.tokenizer_text_limit,
        char_budget=args.tokenizer_char_budget,
    )
    train_char_count = sum(len(text) for text in train_texts)
    print(
        "[stage] training tokenizer "
        f"kind={args.tokenizer_kind} "
        f"texts={len(train_texts)}/{raw_train_text_count} "
        f"chars={train_char_count}/{raw_train_char_count} "
        f"vocab_size={args.vocab_size} min_pair_freq={args.min_pair_freq}",
        flush=True,
    )
    if args.tokenizer_kind == "hf_bpe":
        tokenizer = HuggingFaceBpeTokenizer.train(
            train_texts,
            vocab_size=args.vocab_size,
            min_frequency=args.min_pair_freq,
        )
    else:
        tokenizer = LearnedBpeTokenizer.train(
            train_texts,
            vocab_size=args.vocab_size,
            min_pair_freq=args.min_pair_freq,
            max_text_chars=args.max_text_chars,
        )
    print(f"[stage] tokenizer ready vocab={len(tokenizer.vocab)} elapsed={time.perf_counter() - started:.2f}s", flush=True)
    print("[stage] building token graphs", flush=True)
    rng = random.Random(args.seed)
    split_indices = list(range(len(rows)))
    rng.shuffle(split_indices)
    val_count_target = max(1, int(len(rows) * args.val_ratio)) if len(rows) > 5 else min(1, len(rows))
    val_indices = set(split_indices[:val_count_target])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tokenizer.json").write_text(json.dumps(tokenizer.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    train_count = 0
    val_count = 0
    annotation_count = 0
    with (args.out_dir / "train.base.jsonl").open("w", encoding="utf-8") as train_f, (
        args.out_dir / "val.base.jsonl"
    ).open("w", encoding="utf-8") as val_f, (args.out_dir / "annotation_input.jsonl").open("w", encoding="utf-8") as ann_f:
        for row_index, row in enumerate(rows):
            graph = build_graph(
                row,
                tokenizer,
                max_query_tokens=args.max_query_tokens,
                max_context_tokens=args.max_context_tokens,
                max_unit_tokens=args.max_unit_tokens,
                max_segments=args.max_segments,
                max_same_piece_edges=args.max_same_piece_edges,
                max_overlap_edges=args.max_overlap_edges,
            )
            if row_index in val_indices:
                val_f.write(json.dumps(graph, ensure_ascii=False) + "\n")
                val_count += 1
            else:
                train_f.write(json.dumps(graph, ensure_ascii=False) + "\n")
                train_count += 1
            ann_f.write(
                json.dumps(
                    {
                        "sample_id": graph["sample_id"],
                        "source": graph["source"],
                        "query": graph["query"],
                        "answer": graph["answer"],
                        "target_text": graph["target_text"],
                        "segments": _segments_for_annotation(row, max_segments=args.max_segments, max_segment_chars=800),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            annotation_count += 1
            if args.progress_every and (row_index + 1 == 1 or (row_index + 1) % args.progress_every == 0):
                print(
                    f"[stage] built graph {row_index + 1}/{len(rows)} "
                    f"train={train_count} val={val_count} elapsed={time.perf_counter() - started:.2f}s",
                    flush=True,
                )
    print(
        f"[stage] built graphs={train_count + val_count} train={train_count} val={val_count} "
        f"elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )
    manifest = {
        "dataset_version": "native_token_reasoning_graph_v3_base",
        "input_jsonl": str(args.input_jsonl),
        "train_count": train_count,
        "val_count": val_count,
        "annotation_count": annotation_count,
        "vocab_size": len(tokenizer.vocab),
        "node_type_vocab": NODE_TYPES,
        "edge_type_vocab": EDGE_TYPES,
        "requires_teacher_annotation": True,
        "schema_note": "Reads token_graph_corpus_v2 fields first: source_segments, text_units, target_text.",
        "streaming_builder": True,
        "tokenizer_train_text_count": len(train_texts),
        "tokenizer_kind": args.tokenizer_kind,
        "tokenizer_raw_text_count": raw_train_text_count,
        "tokenizer_train_char_count": train_char_count,
        "tokenizer_raw_char_count": raw_train_char_count,
        "tokenizer_text_limit": args.tokenizer_text_limit,
        "tokenizer_char_budget": args.tokenizer_char_budget,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
