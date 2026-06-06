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


def split_units(text: str, *, max_units: int) -> list[str]:
    units = [part.strip() for part in SENTENCE_RE.split(text or "") if part.strip()]
    if not units and text.strip():
        units = [text.strip()]
    return units[:max_units]


def trim_words(text: str, *, max_words: int) -> str:
    matches = list(WORD_RE.finditer(text or ""))
    if not matches or len(matches) <= max_words:
        return (text or "").strip()
    end = matches[max_words - 1].end()
    return text[:end].strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def choices_to_text(choices: Any) -> tuple[str, dict[str, str]]:
    labels: list[str] = []
    texts: list[str] = []
    if isinstance(choices, dict):
        labels = [str(x) for x in choices.get("label", []) or []]
        texts = [str(x) for x in choices.get("text", []) or []]
    if not labels and isinstance(choices, list):
        for idx, item in enumerate(choices):
            if isinstance(item, dict):
                labels.append(str(item.get("label", chr(ord("A") + idx))))
                texts.append(str(item.get("text", "")))
    mapping = {label.strip(): normalize_space(text) for label, text in zip(labels, texts)}
    rendered = "\n".join(f"{label}. {text}" for label, text in mapping.items() if text)
    return rendered, mapping


def answer_from_choices(answer_key: Any, mapping: dict[str, str]) -> str:
    key = str(answer_key or "").strip()
    if key in mapping:
        return mapping[key]
    upper = key.upper()
    if upper in mapping:
        return mapping[upper]
    return key


def make_record(
    *,
    sample_id: str,
    source: str,
    task_family: str,
    query: str,
    target_text: str,
    split: str,
    context: str = "",
    choices: str = "",
    source_dataset: str = "",
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    if context.strip():
        segments.append(
            {
                "segment_id": "context",
                "text": trim_words(context, max_words=320),
                "segment_type": "context",
                "timestamp": None,
                "source_port": source,
            }
        )
    if choices.strip():
        segments.append(
            {
                "segment_id": "choices",
                "text": trim_words(choices, max_words=160),
                "segment_type": "choices",
                "timestamp": None,
                "source_port": source,
            }
        )
    if not segments:
        segments.append(
            {
                "segment_id": "question",
                "text": trim_words(query, max_words=160),
                "segment_type": "question",
                "timestamp": None,
                "source_port": source,
            }
        )

    text_units: list[dict[str, Any]] = []
    for segment in segments:
        for idx, unit in enumerate(split_units(segment["text"], max_units=16)):
            text_units.append(
                {
                    "unit_id": f"{segment['segment_id']}:u{idx + 1}",
                    "parent_segment_id": segment["segment_id"],
                    "text": unit,
                    "unit_type": segment["segment_type"],
                    "source_segment_type": segment["segment_type"],
                }
            )

    return {
        "schema_version": "token_graph_corpus_v2",
        "sample_id": sample_id,
        "source": source,
        "split": split,
        "task_family": task_family,
        "query": query,
        "question_date": None,
        "source_segments": segments,
        "text_units": text_units,
        "support_segment_ids": [segment["segment_id"] for segment in segments],
        "support_text_unit_ids": [unit["unit_id"] for unit in text_units[:16]],
        "support_alignment": [],
        "target_text": target_text.strip(),
        "target_tokens": WORD_RE.findall(target_text),
        "corpus_note": "open_source_general_ability",
        "packaging_note": "schema2_graph_packaging_from_hf_general_ability",
        "source_dataset": source_dataset,
    }


def iter_dataset(dataset_name: str, config: str | None, split: str, *, streaming: bool) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    if config:
        ds = load_dataset(dataset_name, config, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)
    for row in ds:
        yield dict(row)


def build_squad(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    question = normalize_space(row.get("question", ""))
    context = normalize_space(row.get("context", ""))
    answers = row.get("answers") or {}
    texts = answers.get("text") if isinstance(answers, dict) else None
    answer = normalize_space(texts[0] if texts else "")
    if not question or not context or not answer:
        return None
    query = f"Question: {question}\nUse the context to answer in one short sentence."
    return query, answer, context, ""


def build_boolq(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    question = normalize_space(row.get("question", ""))
    passage = normalize_space(row.get("passage", ""))
    if not question or not passage or row.get("answer") is None:
        return None
    answer = "Yes." if bool(row.get("answer")) else "No."
    query = f"Question: {question}\nAnswer yes or no using the passage."
    return query, answer, passage, ""


def build_gsm8k(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    question = normalize_space(row.get("question", ""))
    raw_answer = str(row.get("answer", "") or "")
    if not question or not raw_answer:
        return None
    final = raw_answer.split("####")[-1].strip() if "####" in raw_answer else raw_answer.strip()
    final = normalize_space(final)
    if not final:
        return None
    query = f"Question: {question}\nAnswer with the final number first, then a brief explanation."
    return query, final, "", ""


def build_multiple_choice(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    question = normalize_space(row.get("question", "") or row.get("question_stem", ""))
    choices, mapping = choices_to_text(row.get("choices"))
    answer = answer_from_choices(row.get("answerKey") or row.get("answer"), mapping)
    if not question or not choices or not answer:
        return None
    query = f"Question: {question}\nChoose the best answer from the choices."
    return query, answer, "", choices


def build_sciq(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    question = normalize_space(row.get("question", ""))
    answer = normalize_space(row.get("correct_answer", ""))
    support = normalize_space(row.get("support", ""))
    distractors = [normalize_space(row.get(f"distractor{i}", "")) for i in range(1, 4)]
    choices = "\n".join(f"{idx}. {item}" for idx, item in enumerate([answer] + [d for d in distractors if d], start=1) if item)
    if not question or not answer:
        return None
    query = f"Question: {question}\nAnswer briefly."
    return query, answer, support, choices


DATASET_SPECS = [
    {"key": "squad", "dataset": "squad", "config": None, "split": "train", "builder": build_squad, "quota": 4000},
    {"key": "boolq", "dataset": "boolq", "config": None, "split": "train", "builder": build_boolq, "quota": 4000},
    {"key": "gsm8k", "dataset": "gsm8k", "config": "main", "split": "train", "builder": build_gsm8k, "quota": 4000},
    {"key": "arc_easy", "dataset": "ai2_arc", "config": "ARC-Easy", "split": "train", "builder": build_multiple_choice, "quota": 3000},
    {"key": "arc_challenge", "dataset": "ai2_arc", "config": "ARC-Challenge", "split": "train", "builder": build_multiple_choice, "quota": 2000},
    {"key": "openbookqa", "dataset": "openbookqa", "config": "main", "split": "train", "builder": build_multiple_choice, "quota": 2000},
    {"key": "commonsense_qa", "dataset": "commonsense_qa", "config": None, "split": "train", "builder": build_multiple_choice, "quota": 3000},
    {"key": "sciq", "dataset": "sciq", "config": None, "split": "train", "builder": build_sciq, "quota": 4000},
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-label", default="general_ability")
    parser.add_argument("--max-records", type=int, default=24000)
    parser.add_argument("--quotas", default="")
    parser.add_argument("--val-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--streaming", action="store_true", help="Read Hugging Face datasets with streaming=True.")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    quotas = parse_quotas(args.quotas)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    errors: dict[str, str] = {}
    kept = 0

    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for spec in DATASET_SPECS:
            key = spec["key"]
            quota = quotas.get(key, int(spec["quota"]))
            if quota <= 0:
                continue
            try:
                rows = iter_dataset(str(spec["dataset"]), spec["config"], str(spec["split"]), streaming=args.streaming)
                for row in rows:
                    if counts[key] >= quota or kept >= args.max_records:
                        break
                    parsed = spec["builder"](row)
                    if parsed is None:
                        continue
                    query, answer, context, choices = parsed
                    if not answer or word_count(answer) > 96:
                        continue
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
                        source_dataset=str(spec["dataset"]),
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
        "dataset_label": args.dataset_label,
        "max_records": args.max_records,
        "kept_records": kept,
        "counts": dict(counts),
        "errors": errors,
        "streaming": bool(args.streaming),
        "schema_version": "token_graph_corpus_v2",
        "note": "Open-source general ability QA/common-sense/math/instruction corpus converted to token graph schema2.",
    }
    (args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
