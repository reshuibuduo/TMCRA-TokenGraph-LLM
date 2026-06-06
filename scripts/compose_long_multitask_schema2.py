from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


TASK_TEMPLATES = (
    (
        "long_continuation",
        "Continue the following text in coherent natural language. Keep the topic, entities, and style consistent.\n\n{prefix}",
    ),
    (
        "detailed_answer",
        "Using the context below, write a detailed answer about what follows. Preserve the important facts and sequence.\n\nContext:\n{prefix}",
    ),
    (
        "long_report",
        "Write a complete long-form response based on the following context. Continue the information naturally and coherently.\n\n{prefix}",
    ),
    (
        "sequence_explanation",
        "Explain what happens next from this context in a detailed, ordered way. Keep names, events, and cause-effect links stable.\n\n{prefix}",
    ),
    (
        "article_completion",
        "Complete this passage as a fluent article or explanation. Do not change the topic.\n\n{prefix}",
    ),
    (
        "qa_followup",
        "Answer in detail: what is the next relevant information after this context?\n\n{prefix}",
    ),
)


DEFAULT_SOURCE_QUOTAS = {
    "cnn_dailymail": 16000,
    "narrativeqa": 9000,
    "coqa": 3000,
    "databricks_dolly_15k": 1000,
    "wikitext103": 1000,
}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def parse_source_quotas(value: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid quota item: {item!r}")
        source, quota = item.split("=", 1)
        quotas[source.strip()] = int(quota)
    return quotas


def split_units(text: str, *, max_units: int) -> list[str]:
    units = [part.strip() for part in SENTENCE_RE.split(text or "") if part.strip()]
    return units[:max_units] if units else ([text.strip()] if text.strip() else [])


def cut_by_words(text: str, *, start_word: int, word_len: int) -> str:
    matches = list(WORD_RE.finditer(text or ""))
    if not matches or start_word >= len(matches):
        return ""
    start = matches[max(0, start_word)].start()
    end_index = min(len(matches) - 1, start_word + max(1, word_len) - 1)
    end = matches[end_index].end()
    return text[start:end].strip()


def longest_source_text(row: dict[str, Any]) -> str:
    segments = row.get("source_segments")
    if isinstance(segments, list) and segments:
        texts = [str(segment.get("text", "") or "").strip() for segment in segments if isinstance(segment, dict)]
        texts = [text for text in texts if text]
        if texts:
            return max(texts, key=word_count)
    legacy = row.get("legacy", {}) or {}
    nodes = legacy.get("memory_nodes")
    if isinstance(nodes, list) and nodes:
        texts = [str(node.get("text", "") or "").strip() for node in nodes if isinstance(node, dict)]
        texts = [text for text in texts if text]
        if texts:
            return max(texts, key=word_count)
    return str(row.get("target_text", "") or "").strip()


def source_name(row: dict[str, Any]) -> str:
    return str(row.get("source", "") or "unknown")


def make_record(
    *,
    sample_id: str,
    source: str,
    task_family: str,
    query: str,
    prefix: str,
    target_text: str,
    split: str,
    source_note: str,
) -> dict[str, Any]:
    source_segments = [
        {
            "segment_id": "prompt_prefix",
            "text": prefix,
            "segment_type": "prompt_prefix",
            "timestamp": None,
            "source_port": source,
        }
    ]
    text_units = [
        {
            "unit_id": f"prompt_prefix:u{index + 1}",
            "parent_segment_id": "prompt_prefix",
            "text": unit,
            "unit_type": "prefix_sentence",
            "source_segment_type": "prompt_prefix",
        }
        for index, unit in enumerate(split_units(prefix, max_units=32))
    ]
    support_unit_ids = [unit["unit_id"] for unit in text_units[: min(16, len(text_units))]]
    return {
        "schema_version": "token_graph_corpus_v2",
        "sample_id": sample_id,
        "source": source,
        "split": split,
        "task_family": task_family,
        "query": query,
        "question_date": None,
        "source_segments": source_segments,
        "text_units": text_units,
        "support_segment_ids": ["prompt_prefix"],
        "support_text_unit_ids": support_unit_ids,
        "support_alignment": [],
        "target_text": target_text.strip(),
        "target_tokens": WORD_RE.findall(target_text),
        "corpus_note": source_note,
        "packaging_note": "long_multitask_same_document_graph_packaging",
    }


def record_from_document(
    *,
    row: dict[str, Any],
    rng: random.Random,
    sample_index: int,
    source: str,
    task_index: int,
    start_word: int,
    prefix_word_range: tuple[int, int],
    min_target_words: int,
    max_target_words: int,
    split: str,
) -> dict[str, Any] | None:
    text = longest_source_text(row)
    total_words = word_count(text)
    if total_words < prefix_word_range[1] + min_target_words + 16:
        return None
    prefix_words = rng.randint(prefix_word_range[0], prefix_word_range[1])
    if start_word + prefix_words + min_target_words >= total_words:
        return None
    available_target = total_words - start_word - prefix_words
    target_words = rng.randint(min_target_words, min(max_target_words, available_target))
    prefix = cut_by_words(text, start_word=start_word, word_len=prefix_words)
    target = cut_by_words(text, start_word=start_word + prefix_words, word_len=target_words)
    if word_count(prefix) < prefix_word_range[0] or word_count(target) < min_target_words:
        return None
    task_family, template = TASK_TEMPLATES[task_index % len(TASK_TEMPLATES)]
    query = template.format(prefix=prefix)
    return make_record(
        sample_id=f"longtask_{source}_{task_family}_{sample_index:07d}",
        source=f"longdoc_{source}",
        task_family=task_family,
        query=query,
        prefix=prefix,
        target_text=target,
        split=split,
        source_note="long_multitask_same_document_prefix_target",
    )


def record_from_existing_answer(
    *,
    row: dict[str, Any],
    sample_index: int,
    source: str,
    min_target_words: int,
    max_target_words: int,
    split: str,
) -> dict[str, Any] | None:
    target = str(row.get("target_text", "") or "").strip()
    if word_count(target) < min_target_words:
        return None
    if word_count(target) > max_target_words:
        target = cut_by_words(target, start_word=0, word_len=max_target_words)
    query = str(row.get("query", "") or "").strip()
    if not query:
        prefix = longest_source_text(row)
        query = f"Write a detailed answer based on this context.\n\n{cut_by_words(prefix, start_word=0, word_len=96)}"
    prefix = longest_source_text(row)
    prefix = cut_by_words(prefix, start_word=0, word_len=160) if prefix else query
    return make_record(
        sample_id=f"longtask_{source}_existing_answer_{sample_index:07d}",
        source=f"longanswer_{source}",
        task_family="existing_answer",
        query=query,
        prefix=prefix,
        target_text=target,
        split=split,
        source_note="long_multitask_existing_schema2_answer",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-output-records", type=int, default=30000)
    parser.add_argument("--min-target-words", type=int, default=360)
    parser.add_argument("--max-target-words", type=int, default=620)
    parser.add_argument("--prefix-min-words", type=int, default=64)
    parser.add_argument("--prefix-max-words", type=int, default=128)
    parser.add_argument("--windows-per-document", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--source-quotas",
        default=",".join(f"{source}={quota}" for source, quota in DEFAULT_SOURCE_QUOTAS.items()),
    )
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()

    quotas = parse_source_quotas(args.source_quotas)
    rng = random.Random(args.seed)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    read_rows = 0
    kept = 0
    source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    seen_docs: set[str] = set()

    with args.input_jsonl.open("r", encoding="utf-8") as src, args.out_jsonl.open("w", encoding="utf-8") as out:
        for line in src:
            if args.limit_rows and read_rows >= args.limit_rows:
                break
            read_rows += 1
            if not line.strip():
                continue
            row = json.loads(line)
            source = source_name(row)
            if source not in quotas:
                skipped["source_not_in_quota"] += 1
                continue
            if source_counts[source] >= quotas[source]:
                skipped["source_quota_full"] += 1
                if all(source_counts[s] >= quotas[s] for s in quotas):
                    break
                continue

            text = longest_source_text(row)
            doc_hash = hashlib.sha1(normalize_for_hash(text).encode("utf-8")).hexdigest()
            if doc_hash in seen_docs:
                skipped["duplicate_doc"] += 1
                continue
            seen_docs.add(doc_hash)

            text_words = word_count(text)
            max_start = max(0, text_words - args.prefix_max_words - args.min_target_words - 1)
            made_from_doc = 0
            for _ in range(max(1, args.windows_per_document)):
                if source_counts[source] >= quotas[source] or kept >= args.max_output_records:
                    break
                start_word = 0 if max_start <= 0 else rng.randint(0, max_start)
                split = "val" if rng.random() < args.val_ratio else "train"
                sample_index = source_counts[source] + 1
                record = record_from_document(
                    row=row,
                    rng=rng,
                    sample_index=sample_index,
                    source=source,
                    task_index=kept,
                    start_word=start_word,
                    prefix_word_range=(args.prefix_min_words, args.prefix_max_words),
                    min_target_words=args.min_target_words,
                    max_target_words=args.max_target_words,
                    split=split,
                )
                if record is None:
                    continue
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                source_counts[source] += 1
                task_counts[str(record.get("task_family", ""))] += 1
                kept += 1
                made_from_doc += 1
            if made_from_doc:
                if args.progress_every and read_rows % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "read_rows": read_rows,
                                "kept": kept,
                                "source_counts": dict(source_counts),
                                "task_counts": dict(task_counts),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if kept >= args.max_output_records:
                    break
                continue

            if source_counts[source] < quotas[source]:
                record = record_from_existing_answer(
                    row=row,
                    sample_index=source_counts[source] + 1,
                    source=source,
                    min_target_words=args.min_target_words,
                    max_target_words=args.max_target_words,
                    split="val" if rng.random() < args.val_ratio else "train",
                )
            if record is None:
                skipped["too_short"] += 1
                continue

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            source_counts[source] += 1
            task_counts[str(record.get("task_family", ""))] += 1
            kept += 1
            if kept >= args.max_output_records:
                break
            if args.progress_every and read_rows % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "read_rows": read_rows,
                            "kept": kept,
                            "source_counts": dict(source_counts),
                            "task_counts": dict(task_counts),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "read_rows": read_rows,
        "kept_records": kept,
        "min_target_words": args.min_target_words,
        "max_target_words": args.max_target_words,
        "prefix_min_words": args.prefix_min_words,
        "prefix_max_words": args.prefix_max_words,
        "windows_per_document": args.windows_per_document,
        "source_quotas": quotas,
        "source_counts": dict(source_counts),
        "task_counts": dict(task_counts),
        "skipped": dict(skipped),
        "note": "Builds long multitask rows from same-document prefix/target splits; no unrelated target concatenation.",
    }
    args.out_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
