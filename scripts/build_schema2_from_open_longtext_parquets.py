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
DEFAULT_TEXT_COLUMNS = ("text", "story", "content", "article", "document", "body")
DEFAULT_PROMPT_COLUMNS = ("prompt", "instruction", "question", "title", "topic")


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
    for index in range(table.num_rows):
        yield {name: data[name][index] for name in names}


def parse_csv(value: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or defaults


def parse_config_quotas(value: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    if not value.strip():
        return quotas
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"invalid quota item: {item!r}")
        key, raw = item.split("=", 1)
        quotas[key.strip()] = int(raw)
    return quotas


def parse_config_ints(value: str) -> dict[str, int]:
    return parse_config_quotas(value)


def config_from_path(parquet_root: Path, path: Path) -> str:
    relative = path.relative_to(parquet_root)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return path.parent.name


def first_text(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Last resort: use the longest string field. This keeps the script useful
    # across open corpora with unfamiliar column names without dataset-specific code.
    strings = [value.strip() for value in row.values() if isinstance(value, str) and value.strip()]
    return max(strings, key=word_count) if strings else ""


def first_prompt(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def row_to_query_prefix_target(
    row: dict[str, Any],
    *,
    text_columns: tuple[str, ...],
    prompt_columns: tuple[str, ...],
    prefix_words: int,
    max_target_words: int,
) -> tuple[str, str, str] | None:
    text = first_text(row, text_columns)
    if not text:
        return None
    prompt = first_prompt(row, prompt_columns)
    if prompt and prompt != text and word_count(text) >= 64:
        prefix = cut_words(prompt, start=0, length=prefix_words) or prompt
        target = cut_words(text, start=0, length=max_target_words)
        query = prompt
        return query, prefix, target

    total = word_count(text)
    if total < prefix_words + 64:
        return None
    prefix = cut_words(text, start=0, length=prefix_words)
    target = cut_words(text, start=prefix_words, length=min(max_target_words, total - prefix_words))
    query = f"Continue the following open-source long text in coherent natural language.\n\n{prefix}"
    return query, prefix, target


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
    text_units = [
        {
            "unit_id": f"prompt_prefix:u{idx + 1}",
            "parent_segment_id": "prompt_prefix",
            "text": unit,
            "unit_type": "prefix_sentence",
            "source_segment_type": "prompt_prefix",
        }
        for idx, unit in enumerate(split_units(prefix, max_units=32))
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
        "corpus_note": "open_source_longtext",
        "packaging_note": "schema2_graph_packaging_from_open_parquet",
        "source_file": source_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-label", default="open_longtext")
    parser.add_argument("--max-records", type=int, default=50000)
    parser.add_argument(
        "--skip-records",
        type=int,
        default=0,
        help="Skip this many valid records before writing. This enables non-overlapping shard builds.",
    )
    parser.add_argument("--config-quotas", default="")
    parser.add_argument(
        "--skip-records-per-config",
        default="",
        help="Comma-separated config=N valid-record skips, applied before per-config writing. Example: finewebedu=50000,blbooks=10000.",
    )
    parser.add_argument("--max-read-rows-per-config", type=int, default=0)
    parser.add_argument("--text-columns", default=",".join(DEFAULT_TEXT_COLUMNS))
    parser.add_argument("--prompt-columns", default=",".join(DEFAULT_PROMPT_COLUMNS))
    parser.add_argument("--prefix-words", type=int, default=96)
    parser.add_argument("--min-target-words", type=int, default=320)
    parser.add_argument("--max-target-words", type=int, default=620)
    parser.add_argument("--val-ratio", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    text_columns = parse_csv(args.text_columns, DEFAULT_TEXT_COLUMNS)
    prompt_columns = parse_csv(args.prompt_columns, DEFAULT_PROMPT_COLUMNS)
    quotas = parse_config_quotas(args.config_quotas)
    per_config_skips = parse_config_ints(args.skip_records_per_config)
    parquet_files = sorted(args.parquet_root.rglob("*.parquet"))
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_valid = 0
    skipped_per_config: Counter[str] = Counter()
    read_rows = 0
    counts: Counter[str] = Counter()
    read_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for parquet_path in parquet_files:
            config = config_from_path(args.parquet_root, parquet_path)
            if quotas and config not in quotas:
                continue
            if quotas and counts[config] >= quotas[config]:
                continue
            for row in iter_parquet_rows(parquet_path):
                if quotas and counts[config] >= quotas[config]:
                    break
                if args.max_read_rows_per_config and read_counts[config] >= args.max_read_rows_per_config:
                    skipped[f"{config}:read_row_cap"] += 1
                    break
                read_rows += 1
                read_counts[config] += 1
                parsed = row_to_query_prefix_target(
                    row,
                    text_columns=text_columns,
                    prompt_columns=prompt_columns,
                    prefix_words=args.prefix_words,
                    max_target_words=args.max_target_words,
                )
                if parsed is None:
                    skipped["missing_or_short_text"] += 1
                    continue
                query, prefix, target = parsed
                if word_count(target) < args.min_target_words:
                    skipped["target_too_short"] += 1
                    continue
                if skipped_valid < args.skip_records:
                    skipped_valid += 1
                    if args.progress_every and skipped_valid % args.progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "skipped_valid": skipped_valid,
                                    "target_skip_records": args.skip_records,
                                    "read_rows": read_rows,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                    )
                    continue
                if per_config_skips and skipped_per_config[config] < per_config_skips.get(config, 0):
                    skipped_per_config[config] += 1
                    if args.progress_every and skipped_per_config[config] % args.progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "config": config,
                                    "skipped_per_config": skipped_per_config[config],
                                    "target_skip": per_config_skips.get(config, 0),
                                    "read_rows": read_rows,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    continue
                split = "val" if rng.random() < args.val_ratio else "train"
                sample_id = f"{args.dataset_label}_{config}_{per_config_skips.get(config, args.skip_records) + counts[config] + 1:08d}"
                source = f"{args.dataset_label}_{config}"
                record = make_record(
                    sample_id=sample_id,
                    source=source,
                    task_family=source,
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
        "dataset_label": args.dataset_label,
        "parquet_files": [str(path) for path in parquet_files],
        "read_rows": read_rows,
        "read_counts": dict(read_counts),
        "kept_records": kept,
        "skip_records": args.skip_records,
        "skipped_valid": skipped_valid,
        "skip_records_per_config": per_config_skips,
        "skipped_per_config": dict(skipped_per_config),
        "counts": dict(counts),
        "config_quotas": quotas,
        "skipped": dict(skipped),
        "text_columns": text_columns,
        "prompt_columns": prompt_columns,
        "prefix_words": args.prefix_words,
        "min_target_words": args.min_target_words,
        "max_target_words": args.max_target_words,
        "note": "Packages open-source long-text parquet rows into schema2 graph-training rows.",
    }
    args.out_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
