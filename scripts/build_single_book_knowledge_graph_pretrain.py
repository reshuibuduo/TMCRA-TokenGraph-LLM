from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def compact_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_term(text: Any, *, fallback: str = "unknown") -> str:
    value = compact_space(text).strip(" .,:;!?\"'()[]{}")
    return value if value else fallback


def clean_quote(text: Any) -> str:
    return compact_space(text).strip(" \"'")


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def is_uninformative_concept_quote(term: str, quote: str) -> bool:
    term_norm = norm(term)
    quote_norm = norm(quote)
    if not term_norm or not quote_norm:
        return True
    if term_norm == quote_norm:
        return True
    quote_words = [word for word in quote_norm.split() if word not in {"a", "an", "the", "this", "that"}]
    term_words = term_norm.split()
    if quote_words == term_words:
        return True
    if len(quote_words) <= len(term_words) + 1 and all(word in set(term_words) for word in quote_words):
        return True
    return len(quote_norm) < max(8, len(term_norm) + 3)


def load_jsonl(path: Path, *, limit: int = 0) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield json.loads(line)
            if limit and index >= limit:
                break


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def find_relation_edge(row: dict[str, Any], relation: dict[str, Any]) -> dict[str, Any] | None:
    rel_name = str(relation.get("relation") or "").lower()
    quote_norm = norm(relation.get("quote", ""))
    best: dict[str, Any] | None = None
    best_score = -1
    for edge in row.get("knowledge_edges") or []:
        if str(edge.get("edge_type") or "") != "knowledge_relation":
            continue
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
        for raw in edge.get(key) or []:
            value = str(raw)
            if value and value not in out:
                out.append(value)
    for key in ("src", "dst"):
        value = str(edge.get(key) or "")
        if value and value not in out:
            out.append(value)
    return out


def token_ids_for_term(row: dict[str, Any], term: str, *, max_ids: int = 24) -> list[str]:
    term_l = norm(term)
    if not term_l:
        return []
    out: list[str] = []
    for item in row.get("knowledge_tokens") or []:
        token_text = norm(item.get("text", "") or item.get("token", ""))
        if not token_text:
            continue
        if token_text in term_l or term_l in token_text:
            token_id = str(item.get("token_id") or item.get("id") or "")
            if token_id and token_id not in out:
                out.append(token_id)
                if len(out) >= max_ids:
                    break
    return out


def relation_terms_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_terms = {norm(left.get("src", "")), norm(left.get("dst", ""))}
    right_terms = {norm(right.get("src", "")), norm(right.get("dst", ""))}
    left_terms.discard("")
    right_terms.discard("")
    if left_terms & right_terms:
        return True
    for a in left_terms:
        for b in right_terms:
            if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
                return True
    return False


def make_row(
    *,
    source_row: dict[str, Any],
    sample_id: str,
    task_family: str,
    query: str,
    target_text: str,
    support_knowledge_token_ids: list[str],
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "token_graph_knowledge_pretrain_v1",
        "sample_id": sample_id,
        "source": "project_gutenberg_single_book_astronomy_graph_pretrain",
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
        "target_tokens": re.findall(r"\S+|\s+", target_text),
        "knowledge_training_note": (
            "Graph-pretrain target derived from prefix-only book concepts and relation edges. "
            "This is not a question-answer conversion; it trains relation completion, path continuation, "
            "concept grounding, and relation-label recovery over the token knowledge graph."
        ),
        "source_sample_id": source_row.get("sample_id", ""),
        "source_payload": source_payload,
    }


def build_relation_examples(row: dict[str, Any], *, max_relations: int) -> list[dict[str, Any]]:
    relations = list(((row.get("llm_preprocess") or {}).get("relations") or []))
    out: list[dict[str, Any]] = []
    used = 0
    for rel_index, relation in enumerate(relations):
        rel_type = clean_term(relation.get("relation"), fallback="related_to").lower()
        src = clean_term(relation.get("src"), fallback="source concept")
        dst = clean_term(relation.get("dst"), fallback="target concept")
        quote = clean_quote(relation.get("quote"))
        if not quote or len(norm(src)) < 2 or len(norm(dst)) < 2:
            continue
        edge = find_relation_edge(row, relation)
        support_ids = support_ids_from_edge(edge)
        if not support_ids:
            continue
        prefix = f"{row.get('sample_id')}:kgrel:{rel_index:02d}"
        payload = {"relation": relation, "edge": edge}
        out.append(
            make_row(
                source_row=row,
                sample_id=f"{prefix}:statement",
                task_family=f"kg_relation_statement:{rel_type}",
                query=f"GRAPH_RELATION_STATEMENT | source={src} | relation={rel_type} | target={dst}",
                target_text=f"{src} --{rel_type}--> {dst}. Evidence: {quote}",
                support_knowledge_token_ids=support_ids,
                source_payload=payload,
            )
        )
        out.append(
            make_row(
                source_row=row,
                sample_id=f"{prefix}:completion",
                task_family=f"kg_relation_completion:{rel_type}",
                query=f"GRAPH_RELATION_COMPLETION | source={src} | relation={rel_type}",
                target_text=f"{dst}. Evidence path: {quote}",
                support_knowledge_token_ids=support_ids,
                source_payload=payload,
            )
        )
        out.append(
            make_row(
                source_row=row,
                sample_id=f"{prefix}:label",
                task_family=f"kg_relation_label:{rel_type}",
                query=f"GRAPH_EDGE_LABEL | source={src} | target={dst}",
                target_text=f"{rel_type}. Evidence: {quote}",
                support_knowledge_token_ids=support_ids,
                source_payload=payload,
            )
        )
        used += 1
        if max_relations and used >= max_relations:
            break
    return out


def build_concept_examples(row: dict[str, Any], *, max_concepts: int) -> list[dict[str, Any]]:
    concepts = list(((row.get("llm_preprocess") or {}).get("concepts") or []))
    relations = list(((row.get("llm_preprocess") or {}).get("relations") or []))
    out: list[dict[str, Any]] = []
    used = 0
    for concept_index, concept in enumerate(concepts):
        term = clean_term(concept.get("term"), fallback="")
        role = clean_term(concept.get("role"), fallback="concept").lower()
        quote = clean_quote(concept.get("quote"))
        if not term or not quote:
            continue
        if is_uninformative_concept_quote(term, quote):
            continue
        support_ids = token_ids_for_term(row, term)
        linked_relations = []
        for relation in relations:
            if norm(term) and norm(term) in {norm(relation.get("src", "")), norm(relation.get("dst", ""))}:
                linked_relations.append(relation)
                edge = find_relation_edge(row, relation)
                for token_id in support_ids_from_edge(edge)[:24]:
                    if token_id not in support_ids:
                        support_ids.append(token_id)
        if not support_ids:
            continue
        out.append(
            make_row(
                source_row=row,
                sample_id=f"{row.get('sample_id')}:kgconcept:{concept_index:02d}:grounding",
                task_family=f"kg_concept_grounding:{role}",
                query=f"GRAPH_CONCEPT_GROUNDING | concept={term} | role={role}",
                target_text=f"{term}: {quote}",
                support_knowledge_token_ids=support_ids,
                source_payload={"concept": concept, "linked_relations": linked_relations[:3]},
            )
        )
        used += 1
        if max_concepts and used >= max_concepts:
            break
    return out


def build_path_examples(row: dict[str, Any], *, max_paths: int) -> list[dict[str, Any]]:
    relations = list(((row.get("llm_preprocess") or {}).get("relations") or []))
    out: list[dict[str, Any]] = []
    for i, left in enumerate(relations):
        for j, right in enumerate(relations):
            if i >= j:
                continue
            if not relation_terms_overlap(left, right):
                continue
            left_edge = find_relation_edge(row, left)
            right_edge = find_relation_edge(row, right)
            support_ids = support_ids_from_edge(left_edge)
            for token_id in support_ids_from_edge(right_edge):
                if token_id not in support_ids:
                    support_ids.append(token_id)
            if not support_ids:
                continue
            left_rel = clean_term(left.get("relation"), fallback="related_to").lower()
            right_rel = clean_term(right.get("relation"), fallback="related_to").lower()
            left_src = clean_term(left.get("src"), fallback="source concept")
            left_dst = clean_term(left.get("dst"), fallback="middle concept")
            right_src = clean_term(right.get("src"), fallback=left_dst)
            right_dst = clean_term(right.get("dst"), fallback="target concept")
            left_quote = clean_quote(left.get("quote"))
            right_quote = clean_quote(right.get("quote"))
            if not left_quote or not right_quote:
                continue
            out.append(
                make_row(
                    source_row=row,
                    sample_id=f"{row.get('sample_id')}:kgpath:{i:02d}_{j:02d}",
                    task_family="kg_path_continuation",
                    query=(
                        "GRAPH_PATH_CONTINUATION | "
                        f"start={left_src} | step1={left_rel} | bridge={left_dst or right_src} | step2={right_rel}"
                    ),
                    target_text=(
                        f"{left_src} --{left_rel}--> {left_dst}; "
                        f"{right_src} --{right_rel}--> {right_dst}. "
                        f"Evidence: {left_quote} | {right_quote}"
                    ),
                    support_knowledge_token_ids=support_ids,
                    source_payload={"left_relation": left, "right_relation": right, "left_edge": left_edge, "right_edge": right_edge},
                )
            )
            if max_paths and len(out) >= max_paths:
                return out
    return out


def build_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_relations_per_source: int,
    max_concepts_per_source: int,
    max_paths_per_source: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.extend(build_relation_examples(row, max_relations=max_relations_per_source))
        out.extend(build_concept_examples(row, max_concepts=max_concepts_per_source))
        out.extend(build_path_examples(row, max_paths=max_paths_per_source))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-relations-per-source", type=int, default=5)
    parser.add_argument("--max-concepts-per-source", type=int, default=2)
    parser.add_argument("--max-paths-per-source", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source_rows = list(load_jsonl(args.input_jsonl, limit=args.limit))
    rows = build_rows(
        source_rows,
        max_relations_per_source=args.max_relations_per_source,
        max_concepts_per_source=args.max_concepts_per_source,
        max_paths_per_source=args.max_paths_per_source,
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    count = write_jsonl(args.out_jsonl, rows)
    task_counts = Counter(str(row.get("task_family", "")) for row in rows)
    manifest = {
        "schema_version": "token_graph_knowledge_pretrain_v1_manifest",
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "source_rows": len(source_rows),
        "examples": count,
        "sources_with_examples": len({row.get("source_sample_id") for row in rows}),
        "max_relations_per_source": args.max_relations_per_source,
        "max_concepts_per_source": args.max_concepts_per_source,
        "max_paths_per_source": args.max_paths_per_source,
        "task_counts": dict(task_counts.most_common()),
        "note": "Graph knowledge pretraining data. QA builder is not used as the primary training corpus.",
    }
    args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
