from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def cut_words(text: str, *, start: int, length: int) -> str:
    matches = list(WORD_RE.finditer(text or ""))
    if not matches or start >= len(matches):
        return ""
    begin = matches[max(0, start)].start()
    end_index = min(len(matches) - 1, start + max(1, length) - 1)
    end = matches[end_index].end()
    return text[begin:end].strip()


def split_units(text: str, *, max_units: int) -> list[str]:
    units = [part.strip() for part in SENTENCE_RE.split(text or "") if part.strip()]
    if not units and text.strip():
        units = [text.strip()]
    return units[:max_units]


def iter_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read parquet files") from exc
    table = pq.read_table(path)
    names = table.column_names
    data = table.to_pydict()
    row_count = table.num_rows
    for index in range(row_count):
        yield {name: data[name][index] for name in names}


def make_record(
    *,
    sample_id: str,
    source: str,
    task_family: str,
    query: str,
    prefix: str,
    target_text: str,
    split: str,
    source_file: str,
) -> dict[str, Any]:
    units = split_units(prefix, max_units=32)
    text_units = [
        {
            "unit_id": f"prompt_prefix:u{idx + 1}",
            "parent_segment_id": "prompt_prefix",
            "text": unit,
            "unit_type": "prefix_sentence",
            "source_segment_type": "prompt_prefix",
        }
        for idx, unit in enumerate(units)
    ]
    return {
        "schema_version": "token_graph_corpus_v2",
        "sample_id": sample_id,
        "source": source,
        "split": split,
        "task_family": task_family,
        "query": query,
        "question_date": None,
        "source_segments": [
            {
                "segment_id": "prompt_prefix",
                "text": prefix,
                "segment_type": "prompt_prefix",
                "timestamp": None,
                "source_port": source,
            }
        ],
        "text_units": text_units,
        "support_segment_ids": ["prompt_prefix"],
        "support_text_unit_ids": [unit["unit_id"] for unit in text_units[: min(16, len(text_units))]],
        "support_alignment": [],
        "target_text": target_text.strip(),
        "target_tokens": WORD_RE.findall(target_text),
        "corpus_note": "open_source_cosmopedia_longtext",
        "packaging_note": "schema2_graph_packaging_from_cosmopedia_parquet",
        "source_file": source_file,
    }


def row_to_prompt_text(row: dict[str, Any], config: str) -> tuple[str, str, str] | None:
    text = str(row.get("text", "") or "").strip()
    prompt = str(row.get("prompt", "") or "").strip()
    if not text:
        return None
    if config in {"openstax", "wikihow", "web_samples_v2"} and prompt:
        return prompt, prompt, text
    if prompt:
        return prompt, prompt, text
    # For story-like rows without an instruction, use the beginning of the same
    # document as context and the continuation as target.
    total = word_count(text)
    if total < 520:
        return None
    prefix = cut_words(text, start=0, length=96)
    target = cut_words(text, start=96, length=min(620, total - 96))
    query = f"Continue the following text in coherent natural language.\n\n{prefix}"
    return query, prefix, target


def bounded_target(text: str, *, min_words: int, max_words: int) -> str:
    wc = word_count(text)
    if wc < min_words:
        return ""
    if wc <= max_words:
        return text.strip()
    return cut_words(text, start=0, length=max_words)


def config_from_path(path: Path) -> str:
    # Expected shape: .../parquet/<config>/<filename>.parquet
    return path.parent.name


def parse_config_quotas(value: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    if not value.strip():
        return quotas
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid config quota: {item!r}")
        key, raw = item.split("=", 1)
        quotas[key.strip()] = int(raw)
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=50000)
    parser.add_argument("--config-quotas", default="")
    parser.add_argument("--min-target-words", type=int, default=320)
    parser.add_argument("--max-target-words", type=int, default=620)
    parser.add_argument("--val-ratio", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(args.parquet_root.rglob("*.parquet"))
    quotas = parse_config_quotas(args.config_quotas)
    kept = 0
    read_rows = 0
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for parquet_path in parquet_files:
            config = config_from_path(parquet_path)
            if quotas and config not in quotas:
                continue
            if quotas and counts[config] >= quotas[config]:
                continue
            for row in iter_parquet_rows(parquet_path):
                if quotas and counts[config] >= quotas[config]:
                    break
                read_rows += 1
                parsed = row_to_prompt_text(row, config)
                if parsed is None:
                    skipped["missing_or_short_text"] += 1
                    continue
                query, prefix, target = parsed
                target = bounded_target(target, min_words=args.min_target_words, max_words=args.max_target_words)
                if not target:
                    skipped["target_too_short"] += 1
                    continue
                split = "val" if rng.random() < args.val_ratio else "train"
                sample_id = f"cosmopedia_{config}_{kept + 1:08d}"
                record = make_record(
                    sample_id=sample_id,
                    source=f"cosmopedia_{config}",
                    task_family=f"cosmopedia_{config}",
                    query=query,
                    prefix=prefix,
                    target_text=target,
                    split=split,
                    source_file=str(parquet_path),
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                counts[config] += 1
                if args.progress_every and kept % args.progress_every == 0:
                    print(json.dumps({"kept": kept, "read_rows": read_rows, "counts": dict(counts)}, ensure_ascii=False), flush=True)
                if kept >= args.max_records:
                    break
            if kept >= args.max_records:
                break
            if quotas and all(counts[item] >= quotas[item] for item in quotas):
                break

    manifest = {
        "parquet_root": str(args.parquet_root),
        "out_jsonl": str(args.out_jsonl),
        "parquet_files": [str(path) for path in parquet_files],
        "read_rows": read_rows,
        "kept_records": kept,
        "counts": dict(counts),
        "config_quotas": quotas,
        "skipped": dict(skipped),
        "min_target_words": args.min_target_words,
        "max_target_words": args.max_target_words,
        "note": "Builds schema2 rows from open-source Cosmopedia long-text parquet files.",
    }
    args.out_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
