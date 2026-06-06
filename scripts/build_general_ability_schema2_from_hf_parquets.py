from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from build_general_ability_schema2_from_hf import (
    WORD_RE,
    build_boolq,
    build_gsm8k,
    build_multiple_choice,
    build_sciq,
    build_squad,
    make_record,
    word_count,
)


PARQUET_SPECS = [
    {
        "key": "squad",
        "repo": "squad",
        "file": "plain_text/train-00000-of-00001.parquet",
        "builder": build_squad,
        "quota": 5000,
    },
    {
        "key": "boolq",
        "repo": "boolq",
        "file": "data/train-00000-of-00001.parquet",
        "builder": build_boolq,
        "quota": 5000,
    },
    {
        "key": "gsm8k",
        "repo": "gsm8k",
        "file": "main/train-00000-of-00001.parquet",
        "builder": build_gsm8k,
        "quota": 5000,
    },
    {
        "key": "arc_easy",
        "repo": "ai2_arc",
        "file": "ARC-Easy/train-00000-of-00001.parquet",
        "builder": build_multiple_choice,
        "quota": 3500,
    },
    {
        "key": "arc_challenge",
        "repo": "ai2_arc",
        "file": "ARC-Challenge/train-00000-of-00001.parquet",
        "builder": build_multiple_choice,
        "quota": 2500,
    },
    {
        "key": "openbookqa",
        "repo": "openbookqa",
        "file": "main/train-00000-of-00001.parquet",
        "builder": build_multiple_choice,
        "quota": 2500,
    },
    {
        "key": "commonsense_qa",
        "repo": "commonsense_qa",
        "file": "data/train-00000-of-00001.parquet",
        "builder": build_multiple_choice,
        "quota": 3500,
    },
    {
        "key": "sciq",
        "repo": "sciq",
        "file": "data/train-00000-of-00001.parquet",
        "builder": build_sciq,
        "quota": 5000,
    },
]


def parse_quotas(raw: str) -> dict[str, int]:
    if not raw.strip():
        return {}
    out: dict[str, int] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        out[key.strip()] = int(value)
    return out


def iter_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    table = pq.read_table(path)
    data = table.to_pydict()
    names = table.column_names
    for index in range(table.num_rows):
        yield {name: data[name][index] for name in names}


def format_target_text(task_family: str, answer: str, *, target_style: str) -> str:
    answer = " ".join(str(answer or "").split()).strip()
    if target_style == "raw":
        return answer
    if not answer:
        return answer
    if task_family in {"arc_easy", "arc_challenge", "openbookqa", "commonsense_qa"}:
        return f"The best answer is {answer}."
    if task_family == "gsm8k":
        return f"The answer is {answer}."
    if task_family == "boolq":
        return f"The answer is {answer}"
    return f"The answer is {answer}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-label", default="general_ability")
    parser.add_argument("--max-records", type=int, default=28000)
    parser.add_argument("--quotas", default="")
    parser.add_argument("--val-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument(
        "--target-style",
        choices=["raw", "answer_sentence"],
        default="raw",
        help="Use raw short labels or wrap them as short answer sentences.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    quotas = parse_quotas(args.quotas)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    errors: dict[str, str] = {}
    downloaded: dict[str, str] = {}
    kept = 0

    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for spec in PARQUET_SPECS:
            key = spec["key"]
            quota = quotas.get(key, int(spec["quota"]))
            if quota <= 0:
                continue
            try:
                local_path = Path(
                    hf_hub_download(
                        repo_id=str(spec["repo"]),
                        filename=str(spec["file"]),
                        repo_type="dataset",
                        cache_dir=str(args.cache_dir),
                    )
                )
                downloaded[key] = str(local_path)
                for row in iter_parquet_rows(local_path):
                    if counts[key] >= quota or kept >= args.max_records:
                        break
                    parsed = spec["builder"](row)
                    if parsed is None:
                        continue
                    query, answer, context, choices = parsed
                    if not answer or word_count(answer) > 96:
                        continue
                    answer = format_target_text(key, answer, target_style=args.target_style)
                    split = "val" if rng.random() < args.val_ratio else "train"
                    source = f"{args.dataset_label}_{key}"
                    record = make_record(
                        sample_id=f"{args.dataset_label}_{key}_{counts[key] + 1:07d}",
                        source=source,
                        task_family=key,
                        query=query,
                        target_text=answer,
                        split=split,
                        context=context,
                        choices=choices,
                        source_dataset=str(spec["repo"]),
                    )
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[key] += 1
                    kept += 1
                    if args.progress_every and kept % args.progress_every == 0:
                        print(json.dumps({"kept": kept, "counts": dict(counts)}, ensure_ascii=False), flush=True)
                    if kept >= args.max_records:
                        break
            except Exception as exc:
                errors[key] = repr(exc)
            if kept >= args.max_records:
                break

    manifest = {
        "out_jsonl": str(args.out_jsonl),
        "cache_dir": str(args.cache_dir),
        "dataset_label": args.dataset_label,
        "max_records": args.max_records,
        "kept_records": kept,
        "counts": dict(counts),
        "errors": errors,
        "downloaded": downloaded,
        "target_style": args.target_style,
        "schema_version": "token_graph_corpus_v2",
        "note": "Open-source general ability QA/common-sense/math/instruction parquet corpus converted to token graph schema2.",
    }
    args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
