from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


DEFAULT_GROUPABLE_SOURCES = {
    "tinystories",
    "wikitext103",
    "cnn_dailymail",
    "databricks_dolly_15k",
    "stanford_alpaca",
}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def split_prefix_target(text: str, *, prefix_words: int) -> tuple[str, str]:
    matches = list(WORD_RE.finditer(text or ""))
    if len(matches) <= prefix_words + 1:
        return text.strip(), ""
    cut = matches[prefix_words - 1].end()
    return text[:cut].strip(), text[cut:].strip()


def split_units(text: str, *, max_units: int) -> list[str]:
    units = [part.strip() for part in SENTENCE_RE.split(text or "") if part.strip()]
    return units[:max_units] if units else ([text.strip()] if text.strip() else [])


def make_record(
    *,
    sample_id: str,
    source: str,
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
        for index, unit in enumerate(split_units(prefix, max_units=24))
    ]
    return {
        "schema_version": "token_graph_corpus_v2",
        "sample_id": sample_id,
        "source": source,
        "split": split,
        "query": query,
        "question_date": None,
        "source_segments": source_segments,
        "text_units": text_units,
        "support_segment_ids": ["prompt_prefix"],
        "support_text_unit_ids": [unit["unit_id"] for unit in text_units[:3]],
        "support_alignment": [],
        "target_text": target_text.strip(),
        "target_tokens": WORD_RE.findall(target_text),
        "corpus_note": source_note,
        "packaging_note": "long_target_continuation_graph_packaging",
    }


def flush_buffer(
    *,
    buffer: list[str],
    source: str,
    sample_index: int,
    min_target_words: int,
    max_target_words: int,
    prefix_words: int,
    split: str,
) -> dict[str, Any] | None:
    text = "\n\n".join(part.strip() for part in buffer if part.strip()).strip()
    if word_count(text) < min_target_words + prefix_words:
        return None
    prefix, target = split_prefix_target(text, prefix_words=prefix_words)
    target_words = WORD_RE.findall(target)
    if len(target_words) > max_target_words:
        # Cut by word boundary while preserving original spacing before the cutoff.
        matches = list(WORD_RE.finditer(target))
        target = target[: matches[max_target_words - 1].end()].strip()
    if word_count(target) < min_target_words:
        return None
    query = f"Continue the following text in coherent natural language:\n{prefix}"
    return make_record(
        sample_id=f"longmix_{source}_{sample_index:07d}",
        source=f"longmix_{source}",
        query=query,
        prefix=prefix,
        target_text=target,
        split=split,
        source_note=f"long_target_continuation_from_{source}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-output-records", type=int, default=30000)
    parser.add_argument("--min-target-words", type=int, default=380)
    parser.add_argument("--max-target-words", type=int, default=620)
    parser.add_argument("--prefix-words", type=int, default=48)
    parser.add_argument("--max-buffer-words", type=int, default=760)
    parser.add_argument("--val-ratio", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--groupable-sources", default=",".join(sorted(DEFAULT_GROUPABLE_SOURCES)))
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()

    groupable = {item.strip() for item in args.groupable_sources.split(",") if item.strip()}
    rng = random.Random(args.seed)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    buffers: dict[str, list[str]] = defaultdict(list)
    buffer_words: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    kept = 0
    read_rows = 0
    with args.input_jsonl.open("r", encoding="utf-8") as src, args.out_jsonl.open("w", encoding="utf-8") as out:
        for line in src:
            if args.limit_rows and read_rows >= args.limit_rows:
                break
            read_rows += 1
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source", "") or "unknown")
            if source not in groupable:
                continue
            text = str(row.get("target_text", "") or "").strip()
            wc = word_count(text)
            if wc < 20:
                continue
            buffers[source].append(text)
            buffer_words[source] += wc
            if buffer_words[source] < args.max_buffer_words:
                continue
            source_counts[source] += 1
            record = flush_buffer(
                buffer=buffers[source],
                source=source,
                sample_index=source_counts[source],
                min_target_words=args.min_target_words,
                max_target_words=args.max_target_words,
                prefix_words=args.prefix_words,
                split="val" if rng.random() < args.val_ratio else "train",
            )
            buffers[source] = []
            buffer_words[source] = 0
            if record is None:
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= args.max_output_records:
                break
            if args.progress_every and read_rows % args.progress_every == 0:
                print(json.dumps({"read_rows": read_rows, "kept": kept}, ensure_ascii=False), flush=True)

        for source, buffer in list(buffers.items()):
            if kept >= args.max_output_records:
                break
            source_counts[source] += 1
            record = flush_buffer(
                buffer=buffer,
                source=source,
                sample_index=source_counts[source],
                min_target_words=args.min_target_words,
                max_target_words=args.max_target_words,
                prefix_words=args.prefix_words,
                split="val" if rng.random() < args.val_ratio else "train",
            )
            if record is None:
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "read_rows": read_rows,
        "kept_records": kept,
        "min_target_words": args.min_target_words,
        "max_target_words": args.max_target_words,
        "prefix_words": args.prefix_words,
        "max_buffer_words": args.max_buffer_words,
        "groupable_sources": sorted(groupable),
    }
    args.out_jsonl.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
