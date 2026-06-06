from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RELATION_TEMPLATES = {
    "defines": (
        "In A Text-Book of Astronomy, what does the passage define or name as {dst}?",
        "The passage defines or names {dst} in relation to {src}. Evidence: {quote}",
    ),
    "causes": (
        "In A Text-Book of Astronomy, what cause or explanation connects {src} with {dst}?",
        "The passage gives this causal explanation: {quote}",
    ),
    "contrasts": (
        "How does A Text-Book of Astronomy contrast {src} with {dst}?",
        "The contrast is: {quote}",
    ),
    "measures": (
        "What measurement relationship does A Text-Book of Astronomy state about {src} and {dst}?",
        "The measurement relationship is: {quote}",
    ),
    "uses": (
        "How is {src} used in the astronomy passage?",
        "The passage states the use as: {quote}",
    ),
    "part_of": (
        "What part-whole relationship does the astronomy passage give between {src} and {dst}?",
        "The part-whole relationship is: {quote}",
    ),
    "example_of": (
        "What example involving {src} and {dst} is given in A Text-Book of Astronomy?",
        "The example is: {quote}",
    ),
    "sequence_next": (
        "What sequence relationship is stated between {src} and {dst} in the astronomy text?",
        "The sequence relationship is: {quote}",
    ),
    "other": (
        "What relationship does the astronomy passage state between {src} and {dst}?",
        "The passage states: {quote}",
    ),
}


CONCEPT_TEMPLATES = {
    "definition": (
        "What does A Text-Book of Astronomy say about {term}?",
        "The passage describes {term} this way: {quote}",
    ),
    "object": (
        "What is {term} in this astronomy passage?",
        "In this passage, {term} is discussed with this evidence: {quote}",
    ),
    "property": (
        "What property or description is associated with {term}?",
        "The property or description associated with {term} is: {quote}",
    ),
    "place": (
        "What place or location is associated with {term}?",
        "The passage associates {term} with this location evidence: {quote}",
    ),
    "process": (
        "What process involving {term} is described?",
        "The process involving {term} is described as: {quote}",
    ),
    "other": (
        "What astronomy knowledge is stated about {term}?",
        "The passage states this about {term}: {quote}",
    ),
}


def load_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def compact_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_term(text: Any, *, fallback: str = "this concept") -> str:
    value = compact_space(text)
    value = value.strip(" .,:;!?\"'()[]{}")
    return value if value else fallback


def clean_quote(text: Any) -> str:
    value = compact_space(text)
    value = value.strip(" \"'")
    return value


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def find_edge_for_relation(row: dict[str, Any], relation: dict[str, Any]) -> dict[str, Any] | None:
    rel_name = str(relation.get("relation") or relation.get("llm_relation") or "").lower()
    quote_norm = norm(relation.get("quote", ""))
    candidates = [
        edge
        for edge in (row.get("knowledge_edges") or [])
        if str(edge.get("edge_type") or "") == "knowledge_relation"
    ]
    best: dict[str, Any] | None = None
    best_score = -1
    for edge in candidates:
        score = 0
        if rel_name and rel_name == str(edge.get("llm_relation") or edge.get("relation") or "").lower():
            score += 3
        edge_quote_norm = norm(edge.get("quote", ""))
        if quote_norm and edge_quote_norm:
            if quote_norm == edge_quote_norm:
                score += 5
            elif quote_norm in edge_quote_norm or edge_quote_norm in quote_norm:
                score += 3
        if score > best_score:
            best = edge
            best_score = score
    return best if best_score > 0 else None


def support_ids_from_edge(edge: dict[str, Any] | None) -> list[str]:
    if not edge:
        return []
    out: list[str] = []
    for key in ("span_token_ids", "src_token_ids", "dst_token_ids"):
        for token_id in edge.get(key) or []:
            value = str(token_id)
            if value and value not in out:
                out.append(value)
    src = str(edge.get("src") or "")
    dst = str(edge.get("dst") or "")
    for value in (src, dst):
        if value and value not in out:
            out.append(value)
    return out


def target_token_texts(row: dict[str, Any], token_ids: list[str]) -> list[str]:
    by_id = {str(item.get("token_id")): str(item.get("text", "")) for item in row.get("knowledge_tokens") or []}
    return [by_id.get(token_id, "") for token_id in token_ids if by_id.get(token_id)]


def make_row(
    *,
    source_row: dict[str, Any],
    sample_id: str,
    task_family: str,
    query: str,
    target_text: str,
    support_knowledge_token_ids: list[str],
    source_relation: dict[str, Any] | None = None,
    source_concept: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_tokens = []
    # Target token strings are metadata only; the graph builder will tokenize
    # target_text with the dataset tokenizer.
    for piece in re.findall(r"\S+|\s+", target_text):
        if piece:
            target_tokens.append(piece)
    return {
        "schema_version": "token_graph_knowledge_qa_v1",
        "sample_id": sample_id,
        "source": "project_gutenberg_single_book_astronomy_knowledge_qa",
        "split": "train",
        "task_family": task_family,
        "book_title": source_row.get("book_title", "A Text-Book of Astronomy"),
        "book_author": source_row.get("book_author", "George C. Comstock"),
        "source_url": source_row.get("source_url", ""),
        "query": query,
        "question_date": None,
        "source_segments": [],
        "text_units": [],
        "knowledge_tokens": source_row.get("knowledge_tokens") or [],
        "knowledge_edges": source_row.get("knowledge_edges") or [],
        "support_segment_ids": [],
        "support_text_unit_ids": [],
        "support_knowledge_token_ids": support_knowledge_token_ids,
        "negative_knowledge_token_ids": [],
        "target_text": target_text,
        "target_tokens": target_tokens,
        "knowledge_training_note": "Question-answer target generated from prefix-only book knowledge graph; target is derived from source prefix relations/concepts, not from continuation text.",
        "source_sample_id": source_row.get("sample_id", ""),
        "source_relation": source_relation,
        "source_concept": source_concept,
    }


def build_relation_examples(row: dict[str, Any], *, max_per_source: int) -> list[dict[str, Any]]:
    relations = list(((row.get("llm_preprocess") or {}).get("relations") or []))
    out: list[dict[str, Any]] = []
    for rel_index, relation in enumerate(relations):
        rel_type = str(relation.get("relation") or "other").lower()
        template = RELATION_TEMPLATES.get(rel_type, RELATION_TEMPLATES["other"])
        src = clean_term(relation.get("src"), fallback="one astronomy concept")
        dst = clean_term(relation.get("dst"), fallback="another astronomy concept")
        quote = clean_quote(relation.get("quote"))
        if not quote or len(norm(quote)) < 8:
            continue
        edge = find_edge_for_relation(row, relation)
        support_ids = support_ids_from_edge(edge)
        if not support_ids:
            continue
        query = template[0].format(src=src, dst=dst, quote=quote)
        target = template[1].format(src=src, dst=dst, quote=quote)
        out.append(
            make_row(
                source_row=row,
                sample_id=f"{row.get('sample_id')}:relqa:{rel_index:02d}",
                task_family=f"single_book_relation_qa:{rel_type}",
                query=query,
                target_text=target,
                support_knowledge_token_ids=support_ids,
                source_relation=relation,
            )
        )
        if max_per_source and len(out) >= max_per_source:
            break
    return out


def build_concept_examples(row: dict[str, Any], *, max_per_source: int) -> list[dict[str, Any]]:
    concepts = list(((row.get("llm_preprocess") or {}).get("concepts") or []))
    relations = list(((row.get("llm_preprocess") or {}).get("relations") or []))
    by_term_quote: dict[str, str] = {}
    by_term_edge: dict[str, dict[str, Any]] = {}
    for relation in relations:
        edge = find_edge_for_relation(row, relation)
        quote = clean_quote(relation.get("quote"))
        if not edge or not quote:
            continue
        for key in ("src", "dst"):
            term = norm(relation.get(key, ""))
            if term and term not in by_term_quote:
                by_term_quote[term] = quote
                by_term_edge[term] = edge
    out: list[dict[str, Any]] = []
    for concept_index, concept in enumerate(concepts):
        term = clean_term(concept.get("term"), fallback="")
        if not term:
            continue
        role = str(concept.get("role") or "other").lower()
        quote = clean_quote(concept.get("quote")) or by_term_quote.get(norm(term), "")
        edge = by_term_edge.get(norm(term))
        support_ids = support_ids_from_edge(edge)
        if not support_ids:
            # Fall back to token ids that literally mention the concept.
            term_n = norm(term)
            support_ids = [
                str(item.get("token_id"))
                for item in row.get("knowledge_tokens") or []
                if term_n and norm(item.get("text", "")) and norm(item.get("text", "")) in term_n
            ][:16]
        if not quote or not support_ids:
            continue
        template = CONCEPT_TEMPLATES.get(role, CONCEPT_TEMPLATES["other"])
        query = template[0].format(term=term, quote=quote)
        target = template[1].format(term=term, quote=quote)
        out.append(
            make_row(
                source_row=row,
                sample_id=f"{row.get('sample_id')}:conceptqa:{concept_index:02d}",
                task_family=f"single_book_concept_qa:{role}",
                query=query,
                target_text=target,
                support_knowledge_token_ids=support_ids,
                source_concept=concept,
            )
        )
        if max_per_source and len(out) >= max_per_source:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-relation-examples-per-source", type=int, default=6)
    parser.add_argument("--max-concept-examples-per-source", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl, limit=args.limit)
    examples: list[dict[str, Any]] = []
    per_source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    for row in rows:
        built = []
        built.extend(build_relation_examples(row, max_per_source=args.max_relation_examples_per_source))
        built.extend(build_concept_examples(row, max_per_source=args.max_concept_examples_per_source))
        for item in built:
            per_source_counts[str(row.get("sample_id", ""))] += 1
            task_counts[str(item.get("task_family", ""))] += 1
        examples.extend(built)

    if args.shuffle:
        random.Random(args.seed).shuffle(examples)
    count = write_jsonl(args.out_jsonl, examples)
    manifest = {
        "dataset_version": "single_book_knowledge_qa_v1",
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "source_rows": len(rows),
        "examples": count,
        "sources_with_examples": len(per_source_counts),
        "min_examples_per_source": min(per_source_counts.values()) if per_source_counts else 0,
        "max_examples_per_source": max(per_source_counts.values()) if per_source_counts else 0,
        "avg_examples_per_source": round(sum(per_source_counts.values()) / max(1, len(per_source_counts)), 3),
        "task_counts": dict(task_counts.most_common()),
        "max_relation_examples_per_source": args.max_relation_examples_per_source,
        "max_concept_examples_per_source": args.max_concept_examples_per_source,
        "note": "All examples are derived from prefix-only LLM preprocessing over a complete public-domain astronomy textbook.",
    }
    manifest_path = args.manifest_json or args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
