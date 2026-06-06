from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from native_token_graph_common import BOS, EOS, HuggingFaceBpeTokenizer, LearnedBpeTokenizer, write_jsonl


NODE_TYPES = {
    "query_token": 1,
    "context_token": 2,
    "unit_token": 3,
    "knowledge_token": 4,
    "knowledge_query_token": 5,
    "knowledge_entity_token": 6,
    "knowledge_relation_token": 7,
    "knowledge_value_token": 8,
    "knowledge_choice_token": 9,
    "knowledge_negative_token": 10,
    "target_prefix_token": 11,
}

EDGE_TYPES = {
    "next_token": 1,
    "prev_token": 2,
    "same_piece": 3,
    "query_context_overlap": 4,
    "query_unit_overlap": 5,
    "answer_overlap_hint": 6,
    "knowledge_next_token": 7,
    "knowledge_prev_token": 8,
    "knowledge_relation": 9,
    "query_knowledge_overlap": 10,
    "support_path": 11,
    "negative_path": 12,
    "target_next_token": 13,
    "target_prev_token": 14,
    "query_target_overlap": 15,
    "context_target_overlap": 16,
    "unit_target_overlap": 17,
    "knowledge_target_overlap": 18,
    # 20-22 are reserved by older teacher-merge outputs.
    "semantic_same_entity": 30,
    "semantic_entity_attribute": 31,
    "semantic_relation": 32,
    "semantic_cause_effect": 33,
    "semantic_condition_result": 34,
    "semantic_temporal": 35,
    "semantic_definition": 36,
    "semantic_example": 37,
    "semantic_contrast": 38,
    "semantic_part_whole": 39,
    "semantic_quantity": 40,
    "semantic_coreference": 41,
    "semantic_support": 42,
    "semantic_negative": 43,
    "semantic_tunnel": 44,
}

STOP_PIECES = {
    "",
    ".",
    ",",
    ":",
    ";",
    "!",
    "?",
    "-",
    "--",
    "—",
    "'",
    '"',
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "by",
    "as",
    "at",
    "from",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _piece_surface(piece: str) -> str:
    return piece.replace("Ġ", "").replace("Ċ", "").strip().lower()


def _token_surface(piece: str) -> str:
    return (
        str(piece or "")
        .replace("臓", "")
        .replace("膴", "")
        .replace("Ġ", "")
        .replace("▁", "")
        .replace("Ċ", "")
        .strip()
        .lower()
    )


def _is_content_piece(piece: str) -> bool:
    surface = _token_surface(piece)
    if surface in STOP_PIECES:
        return False
    if len(surface) <= 1 and not surface.isdigit():
        return False
    return any(ch.isalnum() for ch in surface)


def _mark_support(nodes: list[dict[str, Any]], indices: list[int]) -> None:
    for node_idx in indices:
        if _is_content_piece(str(nodes[node_idx].get("piece", ""))):
            nodes[node_idx]["support_label"] = 1
            nodes[node_idx]["teacher_support_label"] = 1


def _mark_negative(nodes: list[dict[str, Any]], indices: list[int]) -> None:
    for node_idx in indices:
        if _is_content_piece(str(nodes[node_idx].get("piece", ""))):
            nodes[node_idx]["teacher_negative_label"] = 1


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int]] = set()
    out: list[dict[str, Any]] = []
    for edge in edges:
        key = (int(edge["src"]), int(edge["dst"]), int(edge["edge_type_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _semantic_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("semantic_spans"), list):
        return list(row.get("semantic_spans") or [])
    if isinstance(row.get("token_semantic_spans"), list):
        return list(row.get("token_semantic_spans") or [])
    return []


def _semantic_edges(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("semantic_edges"), list):
        return list(row.get("semantic_edges") or [])
    if isinstance(row.get("token_semantic_edges"), list):
        return list(row.get("token_semantic_edges") or [])
    return []


def _semantic_edge_type(raw_role: str) -> str:
    role = str(raw_role or "").strip().lower().replace("-", "_").replace(" ", "_")
    if role in {"same_entity", "same_as", "alias", "entity_alias"}:
        return "semantic_same_entity"
    if role in {"coreference", "co_reference", "pronoun_reference", "refers_to"}:
        return "semantic_coreference"
    if role in {"entity_attribute", "attribute", "property", "has_property", "has_attribute"}:
        return "semantic_entity_attribute"
    if role in {"cause_effect", "causal", "causes", "leads_to", "effect_of"}:
        return "semantic_cause_effect"
    if role in {"condition_result", "if_then", "condition", "requires", "enables"}:
        return "semantic_condition_result"
    if role in {"temporal", "before_after", "before", "after", "sequence", "time_order"}:
        return "semantic_temporal"
    if role in {"definition", "defines", "is_a", "concept_definition"}:
        return "semantic_definition"
    if role in {"example", "example_of", "instance_of", "illustrates"}:
        return "semantic_example"
    if role in {"contrast", "exception", "opposes", "different_from"}:
        return "semantic_contrast"
    if role in {"part_whole", "contains", "part_of", "component_of"}:
        return "semantic_part_whole"
    if role in {"quantity", "numeric", "amount", "count", "sum", "ratio"}:
        return "semantic_quantity"
    if role in {"support", "evidence", "answer_support", "direct_evidence"}:
        return "semantic_support"
    if role in {"negative", "distractor", "contradicts", "not_evidence"}:
        return "semantic_negative"
    if role in {"tunnel", "long_range", "soft_link", "cross_sentence", "cross_paragraph"}:
        return "semantic_tunnel"
    return "semantic_relation"


def collect_texts(rows: list[dict[str, Any]], *, max_text_chars: int) -> list[str]:
    texts: list[str] = []
    for row in rows:
        texts.append(str(row.get("query", "") or "")[:max_text_chars])
        texts.append(_target_text(row)[:max_text_chars])
        for node in _source_segments(row):
            texts.append(str(node.get("text", "") or "")[:max_text_chars])
        for unit in _text_units(row):
            texts.append(str(unit.get("text", "") or unit.get("content", "") or "")[:max_text_chars])
        for item in _knowledge_tokens(row):
            texts.append(str(item.get("text", "") or item.get("token", "") or "")[:max_text_chars])
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


def _negative_segment_ids(row: dict[str, Any]) -> set[str]:
    if row.get("negative_segment_ids") is not None:
        return set(str(x) for x in (row.get("negative_segment_ids", []) or []))
    legacy = row.get("legacy", {}) or {}
    return set(str(x) for x in (legacy.get("negative_ids", []) or []))


def _negative_text_unit_ids(row: dict[str, Any]) -> set[str]:
    if row.get("negative_text_unit_ids") is not None:
        return set(str(x) for x in (row.get("negative_text_unit_ids", []) or []))
    legacy = row.get("legacy", {}) or {}
    return set(str(x) for x in (legacy.get("negative_unit_ids", []) or []))


def _knowledge_tokens(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("knowledge_tokens"), list):
        return list(row.get("knowledge_tokens") or [])
    return []


def _knowledge_edges(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("knowledge_edges"), list):
        return list(row.get("knowledge_edges") or [])
    return []


def _support_knowledge_token_ids(row: dict[str, Any]) -> set[str]:
    if row.get("support_knowledge_token_ids") is not None:
        return set(str(x) for x in (row.get("support_knowledge_token_ids", []) or []))
    return set()


def _negative_knowledge_token_ids(row: dict[str, Any]) -> set[str]:
    if row.get("negative_knowledge_token_ids") is not None:
        return set(str(x) for x in (row.get("negative_knowledge_token_ids", []) or []))
    return set()


def _knowledge_node_type(role: str) -> str:
    role_l = str(role or "").lower()
    if "negative" in role_l or "distractor" in role_l:
        return "knowledge_negative_token"
    if "query" in role_l or "question" in role_l:
        return "knowledge_query_token"
    if "choice" in role_l or "candidate" in role_l:
        return "knowledge_choice_token"
    if any(marker in role_l for marker in ("relation", "predicate", "action", "process", "verb", "operator")):
        return "knowledge_relation_token"
    if any(marker in role_l for marker in ("answer", "value", "attribute", "input", "output", "quantity", "state")):
        return "knowledge_value_token"
    if any(marker in role_l for marker in ("entity", "concept", "subject", "object", "term", "topic")):
        return "knowledge_entity_token"
    return "knowledge_token"


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


def _add_target_prefix_segment(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    tokenizer: LearnedBpeTokenizer,
    sample_id: str,
    pieces: list[str],
    max_tokens: int,
) -> list[int]:
    prefix_pieces = pieces[:max_tokens]
    piece_ids = [tokenizer.vocab.get(piece, tokenizer.vocab.get("<unk>", 3)) for piece in prefix_pieces]
    indices: list[int] = []
    prev: int | None = None
    for pos, (piece, piece_id) in enumerate(zip(prefix_pieces, piece_ids)):
        index = len(nodes)
        indices.append(index)
        nodes.append(
            {
                "id": f"{sample_id}:target_prefix:tok:{pos:04d}",
                "piece": piece,
                "piece_id": int(piece_id),
                "node_type": "target_prefix_token",
                "node_type_id": NODE_TYPES["target_prefix_token"],
                "segment_id": "target_prefix",
                "source_id": "target_prefix",
                "position": pos,
                "text": piece,
                "support_label": 0,
                "answer_overlap_label": 1 if _is_content_piece(piece) else 0,
                "teacher_support_label": 0,
                "teacher_negative_label": 0,
                "target_prefix_position": pos,
                "causal_target_prefix_node": 1,
            }
        )
        if prev is not None:
            edges.append({"src": prev, "dst": index, "edge_type": "target_next_token", "edge_type_id": EDGE_TYPES["target_next_token"], "label": 1})
            edges.append({"src": index, "dst": prev, "edge_type": "target_prev_token", "edge_type_id": EDGE_TYPES["target_prev_token"], "label": 1})
        prev = index
    return indices


def _add_knowledge_item(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    tokenizer: LearnedBpeTokenizer,
    sample_id: str,
    item_id: str,
    role: str,
    text: str,
    piece: str | None = None,
    piece_id: int | None = None,
    max_tokens: int,
    support_ids: set[str],
    negative_ids: set[str],
) -> list[int]:
    node_type = _knowledge_node_type(role)
    if piece is not None and piece_id is not None:
        pieces = [str(piece)]
        piece_ids = [int(piece_id)]
    else:
        pieces = tokenizer.encode_pieces(text, max_tokens=max_tokens)
        piece_ids = [tokenizer.vocab.get(piece, tokenizer.vocab.get("<unk>", 3)) for piece in pieces]
    indices: list[int] = []
    prev: int | None = None
    support_label = 1 if item_id in support_ids else 0
    negative_label = 1 if item_id in negative_ids or node_type == "knowledge_negative_token" else 0
    for pos, (piece, piece_id) in enumerate(zip(pieces, piece_ids)):
        index = len(nodes)
        indices.append(index)
        nodes.append(
            {
                "id": f"{sample_id}:knowledge:{item_id}:tok:{pos:04d}",
                "piece": piece,
                "piece_id": int(piece_id),
                "node_type": node_type,
                "node_type_id": NODE_TYPES[node_type],
                "knowledge_item_id": item_id,
                "knowledge_role": str(role or ""),
                "segment_id": item_id,
                "source_id": item_id,
                "position": pos,
                "text": piece,
                "support_label": support_label,
                "answer_overlap_label": 0,
                "teacher_support_label": support_label,
                "teacher_negative_label": negative_label,
            }
        )
        if prev is not None:
            edges.append(
                {
                    "src": prev,
                    "dst": index,
                    "edge_type": "knowledge_next_token",
                    "edge_type_id": EDGE_TYPES["knowledge_next_token"],
                    "label": 1,
                }
            )
            edges.append(
                {
                    "src": index,
                    "dst": prev,
                    "edge_type": "knowledge_prev_token",
                    "edge_type_id": EDGE_TYPES["knowledge_prev_token"],
                    "label": 1,
                }
            )
        prev = index
    return indices


def _add_directed_edge(
    edges: list[dict[str, Any]],
    src: int,
    dst: int,
    *,
    edge_type: str,
    label: int = 0,
    bidirectional: bool = False,
) -> None:
    edges.append({"src": src, "dst": dst, "edge_type": edge_type, "edge_type_id": EDGE_TYPES[edge_type], "label": int(label)})
    if bidirectional:
        edges.append({"src": dst, "dst": src, "edge_type": edge_type, "edge_type_id": EDGE_TYPES[edge_type], "label": int(label)})


def _knowledge_node_indices(knowledge_index: dict[str, list[int]], token_ids: list[Any], *, max_items: int = 32) -> list[int]:
    indices: list[int] = []
    for raw_id in token_ids[:max_items]:
        item_id = str(raw_id)
        node_indices = knowledge_index.get(item_id) or []
        if node_indices:
            indices.extend(node_indices)
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


def _span_refs(span: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in (
        "segment_id",
        "segment_ref",
        "source_id",
        "source_segment_id",
        "unit_id",
        "text_unit_id",
        "parent_segment_id",
    ):
        value = str(span.get(key, "") or "")
        if value and value not in refs:
            refs.append(value)
    return refs


def _span_candidate_indices(
    span: dict[str, Any],
    *,
    nodes: list[dict[str, Any]],
    segment_index: dict[str, list[int]],
    source_ref_index: dict[str, list[int]],
) -> list[int]:
    candidates: list[int] = []
    for ref in _span_refs(span):
        if ref in segment_index:
            candidates.extend(segment_index[ref])
        if ref in source_ref_index:
            candidates.extend(source_ref_index[ref])
        if f"knowledge:{ref}" in segment_index:
            candidates.extend(segment_index[f"knowledge:{ref}"])
    if not candidates:
        candidates = list(range(len(nodes)))
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in candidates:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def _resolve_semantic_span_nodes(
    span: dict[str, Any],
    *,
    nodes: list[dict[str, Any]],
    segment_index: dict[str, list[int]],
    source_ref_index: dict[str, list[int]],
    max_nodes: int,
) -> list[int]:
    candidates = _span_candidate_indices(span, nodes=nodes, segment_index=segment_index, source_ref_index=source_ref_index)
    start = span.get("token_start")
    end = span.get("token_end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return candidates[max(0, start) : max(0, end)][:max_nodes]

    raw_quote = str(span.get("quote", "") or span.get("text", "") or span.get("surface", "") or "")
    quote = _normalize_text(raw_quote)
    if not quote:
        return [idx for idx in candidates if _is_content_piece(str(nodes[idx].get("piece", "")))][:max_nodes]

    hits: list[int] = []
    for idx in candidates:
        piece = _token_surface(str(nodes[idx].get("piece", "")))
        if piece and piece in quote and _is_content_piece(piece):
            hits.append(idx)
            if len(hits) >= max_nodes:
                break
    if hits:
        return hits

    # Fallback for byte-level BPE pieces: recover useful alphanumeric fragments
    # from the quote and match against token surfaces.
    terms = {term for term in re.findall(r"[a-z0-9]+", quote.lower()) if len(term) > 1}
    for idx in candidates:
        piece = _token_surface(str(nodes[idx].get("piece", "")))
        if piece and piece in terms and _is_content_piece(piece):
            hits.append(idx)
            if len(hits) >= max_nodes:
                break
    return hits


def _semantic_edge_anchor_nodes(nodes: list[dict[str, Any]], indices: list[int], *, max_nodes: int) -> list[int]:
    content = [idx for idx in indices if _is_content_piece(str(nodes[idx].get("piece", "")))]
    pool = content or list(indices)
    if len(pool) <= max_nodes:
        return pool
    if max_nodes <= 1:
        return [pool[0]]
    if max_nodes == 2:
        return [pool[0], pool[-1]]
    mid = pool[len(pool) // 2]
    anchors = [pool[0], mid, pool[-1]]
    for idx in pool:
        if len(anchors) >= max_nodes:
            break
        if idx not in anchors:
            anchors.append(idx)
    return anchors[:max_nodes]


def _add_semantic_graph_supervision(
    *,
    row: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    segment_index: dict[str, list[int]],
    source_ref_index: dict[str, list[int]],
    max_span_nodes: int,
    max_edge_nodes: int,
) -> dict[str, int]:
    span_nodes: dict[str, list[int]] = {}
    support_count = 0
    negative_count = 0
    for raw_idx, span in enumerate(_semantic_spans(row)):
        span_id = str(span.get("span_id", "") or span.get("id", "") or f"semantic_span_{raw_idx:04d}")
        indices = _resolve_semantic_span_nodes(
            span,
            nodes=nodes,
            segment_index=segment_index,
            source_ref_index=source_ref_index,
            max_nodes=max_span_nodes,
        )
        if not indices:
            continue
        span_nodes[span_id] = indices
        role = str(span.get("role", "") or span.get("type", "") or "").lower()
        if any(marker in role for marker in ("support", "evidence", "answer", "definition", "fact")):
            _mark_support(nodes, indices)
            support_count += 1
        if any(marker in role for marker in ("negative", "distractor", "contradict")):
            _mark_negative(nodes, indices)
            negative_count += 1

    edge_count = 0
    for raw_edge in _semantic_edges(row):
        src_id = str(raw_edge.get("src_span_id", "") or raw_edge.get("src", "") or raw_edge.get("source", ""))
        dst_id = str(raw_edge.get("dst_span_id", "") or raw_edge.get("dst", "") or raw_edge.get("target", ""))
        src_nodes = span_nodes.get(src_id)
        dst_nodes = span_nodes.get(dst_id)
        if not src_nodes or not dst_nodes:
            continue
        src_anchors = _semantic_edge_anchor_nodes(nodes, src_nodes, max_nodes=max_edge_nodes)
        dst_anchors = _semantic_edge_anchor_nodes(nodes, dst_nodes, max_nodes=max_edge_nodes)
        edge_type = _semantic_edge_type(str(raw_edge.get("edge_type", "") or raw_edge.get("relation", "") or raw_edge.get("type", "")))
        label = int(raw_edge.get("label", 1) or 0)
        bidirectional = edge_type in {
            "semantic_same_entity",
            "semantic_coreference",
            "semantic_contrast",
            "semantic_tunnel",
        } or bool(raw_edge.get("bidirectional", False))
        for src in src_anchors:
            for dst in dst_anchors:
                if src == dst:
                    continue
                _add_directed_edge(edges, src, dst, edge_type=edge_type, label=label, bidirectional=bidirectional)
                edge_count += 1
    return {
        "semantic_span_count": len(span_nodes),
        "semantic_edge_count": edge_count,
        "semantic_support_span_count": support_count,
        "semantic_negative_span_count": negative_count,
    }


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
    max_knowledge_tokens: int,
    max_knowledge_items: int,
    max_target_tokens: int,
    max_segments: int,
    max_same_piece_edges: int,
    max_overlap_edges: int,
    max_semantic_span_nodes: int,
    max_semantic_edge_nodes: int,
    graph_mode: str = "base",
    max_target_prefix_tokens: int = 0,
    include_answer_overlap_hints: bool = False,
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
    source_ref_index: dict[str, list[int]] = {"query": query_indices}
    source_supports = _support_segment_ids(row)
    source_supports |= _support_text_unit_ids(row)
    source_negatives = _negative_segment_ids(row)
    source_negatives |= _negative_text_unit_ids(row)
    knowledge_supports = _support_knowledge_token_ids(row)
    knowledge_negatives = _negative_knowledge_token_ids(row)
    text_index: dict[str, list[int]] = {}

    for idx, memory in enumerate(_source_segments(row)[:max_segments]):
        segment_id = f"context_{idx:03d}"
        source_id = str(memory.get("segment_id", segment_id))
        text = str(memory.get("text", "") or "")
        text_key = _normalize_text(text)
        if text_key and text_key in text_index:
            segment_index[segment_id] = text_index[text_key]
        else:
            segment_index[segment_id] = _add_segment(
                nodes=nodes,
                edges=edges,
                tokenizer=tokenizer,
                sample_id=sample_id,
                segment_id=segment_id,
                segment_type="context_token",
                source_id=source_id,
                text=text,
                max_tokens=max_context_tokens,
            )
            if text_key:
                text_index[text_key] = segment_index[segment_id]
        source_ref_index[source_id] = segment_index[segment_id]
        source_ref_index[segment_id] = segment_index[segment_id]
        if source_id in source_supports:
            _mark_support(nodes, segment_index[segment_id])
        if source_id in source_negatives:
            _mark_negative(nodes, segment_index[segment_id])

    knowledge_index: dict[str, list[int]] = {}
    for idx, item in enumerate(_knowledge_tokens(row)[:max_knowledge_items]):
        item_id = str(item.get("token_id", "") or item.get("id", "") or f"knowledge_{idx:03d}")
        role = str(item.get("role", "") or item.get("type", "") or "knowledge")
        text = str(item.get("text", "") or item.get("token", "") or "")
        if not text.strip():
            continue
        indices = _add_knowledge_item(
            nodes=nodes,
            edges=edges,
            tokenizer=tokenizer,
            sample_id=sample_id,
            item_id=item_id,
            role=role,
            text=text,
            piece=str(item.get("piece")) if item.get("piece") is not None else None,
            piece_id=int(item.get("piece_id")) if item.get("piece_id") is not None else None,
            max_tokens=max_knowledge_tokens,
            support_ids=knowledge_supports,
            negative_ids=knowledge_negatives,
        )
        if indices:
            knowledge_index[item_id] = indices
            segment_index[f"knowledge:{item_id}"] = indices
            source_ref_index[item_id] = indices

    for edge in _knowledge_edges(row):
        src_id = str(edge.get("src", "") or edge.get("source", "") or edge.get("from", ""))
        dst_id = str(edge.get("dst", "") or edge.get("target", "") or edge.get("to", ""))
        edge_role = str(edge.get("edge_type", "") or edge.get("type", "") or "knowledge_relation").lower()
        if "negative" in edge_role:
            graph_edge_type = "negative_path"
            label = 0
        elif "support" in edge_role or src_id in knowledge_supports or dst_id in knowledge_supports:
            graph_edge_type = "support_path"
            label = 1
        else:
            graph_edge_type = "knowledge_relation"
            label = int(edge.get("label", 0) or 0)
        span_nodes = _knowledge_node_indices(knowledge_index, list(edge.get("span_token_ids") or []), max_items=48)
        if len(span_nodes) >= 2 and "sequence" not in edge_role:
            for left, right in zip(span_nodes, span_nodes[1:]):
                _add_directed_edge(edges, left, right, edge_type=graph_edge_type, label=label, bidirectional=True)
        if not src_id or not dst_id or src_id not in knowledge_index or dst_id not in knowledge_index:
            if len(span_nodes) >= 2 and "sequence" not in edge_role:
                _add_directed_edge(edges, span_nodes[0], span_nodes[-1], edge_type=graph_edge_type, label=label, bidirectional=True)
            continue
        src_nodes = knowledge_index[src_id]
        dst_nodes = knowledge_index[dst_id]
        if not src_nodes or not dst_nodes:
            continue
        _add_directed_edge(edges, src_nodes[-1], dst_nodes[0], edge_type=graph_edge_type, label=label, bidirectional=True)

    for idx, unit in enumerate(_text_units(row)[:max_segments]):
        segment_id = f"unit_{idx:03d}"
        source_id = str(unit.get("unit_id", segment_id))
        text = str(unit.get("text", "") or unit.get("content", "") or "")
        text_key = _normalize_text(text)
        if text_key and text_key in text_index:
            segment_index[segment_id] = text_index[text_key]
        else:
            segment_index[segment_id] = _add_segment(
                nodes=nodes,
                edges=edges,
                tokenizer=tokenizer,
                sample_id=sample_id,
                segment_id=segment_id,
                segment_type="unit_token",
                source_id=source_id,
                text=text,
                max_tokens=max_unit_tokens,
            )
            if text_key:
                text_index[text_key] = segment_index[segment_id]
        source_ref_index[source_id] = segment_index[segment_id]
        source_ref_index[segment_id] = segment_index[segment_id]
        if source_id in source_supports:
            _mark_support(nodes, segment_index[segment_id])
        if source_id in source_negatives:
            _mark_negative(nodes, segment_index[segment_id])

    answer_pieces = tokenizer.encode_pieces(_target_text(row), max_tokens=max_target_tokens)
    answer_ids = [tokenizer.vocab.get(piece, tokenizer.vocab.get("<unk>", 3)) for piece in answer_pieces]
    answer_set = set(answer_ids)
    answer_content_set = {
        int(piece_id)
        for piece, piece_id in zip(answer_pieces, answer_ids)
        if _is_content_piece(piece)
    }
    target_prefix_indices: list[int] = []
    if graph_mode == "simple_plus_causal_target":
        target_prefix_limit = max_target_prefix_tokens if max_target_prefix_tokens > 0 else max_target_tokens
        target_prefix_indices = _add_target_prefix_segment(
            nodes=nodes,
            edges=edges,
            tokenizer=tokenizer,
            sample_id=sample_id,
            pieces=answer_pieces,
            max_tokens=target_prefix_limit,
        )
    for node in nodes:
        if int(node["piece_id"]) in answer_content_set:
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
        if segment_id.startswith("knowledge:"):
            edge_type = "query_knowledge_overlap"
        else:
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
    if target_prefix_indices:
        target_piece_ids = [int(nodes[i]["piece_id"]) for i in target_prefix_indices]
        _bounded_pair_edges(
            edges,
            query_indices,
            target_prefix_indices,
            left_piece_ids=query_piece_ids,
            right_piece_ids=target_piece_ids,
            edge_type="query_target_overlap",
            max_edges=max_overlap_edges,
        )
        for segment_id, indices in segment_index.items():
            if segment_id == "query":
                continue
            piece_ids = [int(nodes[i]["piece_id"]) for i in indices]
            if segment_id.startswith("knowledge:"):
                edge_type = "knowledge_target_overlap"
            elif segment_id.startswith("unit_"):
                edge_type = "unit_target_overlap"
            else:
                edge_type = "context_target_overlap"
            _bounded_pair_edges(
                edges,
                indices,
                target_prefix_indices,
                left_piece_ids=piece_ids,
                right_piece_ids=target_piece_ids,
                edge_type=edge_type,
                max_edges=max_overlap_edges,
            )
    semantic_stats = _add_semantic_graph_supervision(
        row=row,
        nodes=nodes,
        edges=edges,
        segment_index=segment_index,
        source_ref_index=source_ref_index,
        max_span_nodes=max_semantic_span_nodes,
        max_edge_nodes=max_semantic_edge_nodes,
    )
    if include_answer_overlap_hints:
        for idx, node in enumerate(nodes):
            if int(node["piece_id"]) not in answer_content_set:
                continue
            for q_idx in query_indices[:8]:
                edges.append({"src": q_idx, "dst": idx, "edge_type": "answer_overlap_hint", "edge_type_id": EDGE_TYPES["answer_overlap_hint"], "label": 0})

    answer_target_ids = [tokenizer.vocab[BOS]] + answer_ids + [tokenizer.vocab[EOS]]
    edges = _dedupe_edges(edges)
    return {
        "dataset_version": "native_token_reasoning_graph_v3_simple_plus" if graph_mode == "simple_plus_causal_target" else "native_token_reasoning_graph_v3_base",
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
        "semantic_stats": semantic_stats,
        "graph_mode": graph_mode,
        "target_prefix_node_count": len(target_prefix_indices),
        "target_prefix_causal_note": "target_prefix_token nodes are teacher-forcing graph nodes; training must apply causal masking so future target nodes are not used as ordinary evidence." if target_prefix_indices else "",
        "node_type_vocab": NODE_TYPES,
        "edge_type_vocab": EDGE_TYPES,
        "annotation": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--skip-records",
        type=int,
        default=0,
        help="Skip this many source JSONL records before applying --limit. Useful for non-overlapping shard builds.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tokenizer-kind", choices=["hf_pretrained", "hf_bpe", "char_bpe"], default="hf_pretrained")
    parser.add_argument("--pretrained-tokenizer", default="gpt2")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-pair-freq", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=512)
    parser.add_argument("--tokenizer-text-limit", type=int, default=1000)
    parser.add_argument("--tokenizer-char-budget", type=int, default=250000)
    parser.add_argument("--max-query-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=128)
    parser.add_argument("--max-unit-tokens", type=int, default=128)
    parser.add_argument("--max-knowledge-tokens", type=int, default=16)
    parser.add_argument("--max-knowledge-items", type=int, default=96)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument("--max-segments", type=int, default=32)
    parser.add_argument("--max-same-piece-edges", type=int, default=384)
    parser.add_argument("--max-overlap-edges", type=int, default=192)
    parser.add_argument("--max-semantic-span-nodes", type=int, default=12)
    parser.add_argument("--max-semantic-edge-nodes", type=int, default=3)
    parser.add_argument(
        "--graph-mode",
        choices=["base", "simple_plus_causal_target"],
        default="base",
        help="base preserves the previous prompt/context graph. simple_plus_causal_target adds teacher-forcing target-prefix nodes and causal target edges.",
    )
    parser.add_argument(
        "--max-target-prefix-tokens",
        type=int,
        default=0,
        help="Maximum target-prefix nodes to add in simple_plus_causal_target mode. Defaults to max-target-tokens when 0.",
    )
    parser.add_argument(
        "--include-answer-overlap-hints",
        action="store_true",
        help="Add target-derived answer_overlap_hint edges. Disabled by default to align training graphs with inference graphs.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    started = time.perf_counter()
    print(f"[stage] loading rows from {args.input_jsonl}", flush=True)
    load_limit = args.limit + args.skip_records if args.limit else 0
    rows = load_rows(args.input_jsonl, limit=load_limit)
    if args.skip_records:
        rows = rows[args.skip_records :]
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
    if args.tokenizer_kind == "hf_pretrained":
        tokenizer = HuggingFaceBpeTokenizer.from_pretrained_model(args.pretrained_tokenizer)
    elif args.tokenizer_kind == "hf_bpe":
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
                max_knowledge_tokens=args.max_knowledge_tokens,
                max_knowledge_items=args.max_knowledge_items,
                max_target_tokens=args.max_target_tokens,
                max_segments=args.max_segments,
                max_same_piece_edges=args.max_same_piece_edges,
                max_overlap_edges=args.max_overlap_edges,
                max_semantic_span_nodes=args.max_semantic_span_nodes,
                max_semantic_edge_nodes=args.max_semantic_edge_nodes,
                graph_mode=args.graph_mode,
                max_target_prefix_tokens=args.max_target_prefix_tokens,
                include_answer_overlap_hints=args.include_answer_overlap_hints,
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
        "dataset_version": "native_token_reasoning_graph_v3_simple_plus" if args.graph_mode == "simple_plus_causal_target" else "native_token_reasoning_graph_v3_base",
        "input_jsonl": str(args.input_jsonl),
        "skip_records": args.skip_records,
        "limit": args.limit,
        "train_count": train_count,
        "val_count": val_count,
        "annotation_count": annotation_count,
        "vocab_size": len(tokenizer.vocab),
        "node_type_vocab": NODE_TYPES,
        "edge_type_vocab": EDGE_TYPES,
        "requires_teacher_annotation": True,
        "streaming_builder": True,
        "tokenizer_train_text_count": len(train_texts),
        "tokenizer_kind": args.tokenizer_kind,
        "pretrained_tokenizer": args.pretrained_tokenizer if args.tokenizer_kind == "hf_pretrained" else "",
        "tokenizer_raw_text_count": raw_train_text_count,
        "tokenizer_train_char_count": train_char_count,
        "tokenizer_raw_char_count": raw_train_char_count,
        "tokenizer_text_limit": args.tokenizer_text_limit,
        "tokenizer_char_budget": args.tokenizer_char_budget,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "max_target_tokens": args.max_target_tokens,
        "max_knowledge_tokens": args.max_knowledge_tokens,
        "max_knowledge_items": args.max_knowledge_items,
        "max_semantic_span_nodes": args.max_semantic_span_nodes,
        "max_semantic_edge_nodes": args.max_semantic_edge_nodes,
        "graph_mode": args.graph_mode,
        "max_target_prefix_tokens": args.max_target_prefix_tokens,
        "include_answer_overlap_hints": args.include_answer_overlap_hints,
        "schema_note": "Reads token_graph_corpus_v2 fields plus optional token_graph_knowledge_v1 knowledge_tokens/knowledge_edges and token_graph_semantic_v1 semantic_spans/semantic_edges.",
        "graph_alignment_note": "answer_overlap_hint edges are disabled by default because they are target-derived and unavailable at inference time. knowledge_tokens are expanded into BPE-token graph nodes rather than chunk-level memory units.",
        "simple_plus_note": "simple_plus_causal_target adds target_prefix_token nodes and target causal edges for graph-native language modeling. Training must treat these as teacher-forcing prefix nodes with causal masking, not as unrestricted future-answer evidence.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
