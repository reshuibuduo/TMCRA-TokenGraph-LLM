from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable


SOURCE_DEFAULTS = {
    "squad": 87599,
    "dolly": 15000,
    "gsm8k": 7500,
    "tinystories": 100000,
    "wikitext103": 70000,
    "cnn_dailymail": 45000,
}


def write_progress(out_dir: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clipped(text: str, limit: int) -> str:
    return str(text or "").strip()[:limit]


def dump_source(
    *,
    out_dir: Path,
    source: str,
    dataset_name: str,
    config: str | None,
    split: str,
    limit: int,
    mapper,
    streaming: bool,
    progress_every: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    out_path = out_dir / f"{source}.jsonl"
    started = time.perf_counter()
    count = 0
    write_progress(out_dir, {"status": "running", "source": source, "source_count": 0})
    print(f"[source] start {source} dataset={dataset_name} config={config} split={split} limit={limit}", flush=True)

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if config is None:
        ds = load_dataset(dataset_name, **kwargs)
    else:
        ds = load_dataset(dataset_name, config, **kwargs)

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            mapped = mapper(row, i)
            if mapped is None:
                continue
            f.write(json.dumps(mapped, ensure_ascii=False) + "\n")
            count += 1
            if progress_every and (count == 1 or count % progress_every == 0):
                print(f"[source] {source} count={count} elapsed={time.perf_counter() - started:.1f}s", flush=True)
                write_progress(out_dir, {"status": "running", "source": source, "source_count": count})
            if limit > 0 and count >= limit:
                break

    info = {
        "source": source,
        "dataset": dataset_name,
        "config": config,
        "split": split,
        "limit": limit,
        "count": count,
        "path": str(out_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(f"[source] done {source} count={count} elapsed={info['elapsed_seconds']}s", flush=True)
    return info


def map_squad(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    answers = row.get("answers") or {}
    texts = answers.get("text") or []
    answer = clipped(texts[0] if texts else "", 2000)
    context = clipped(row.get("context") or "", 8000)
    question = clipped(row.get("question") or "", 1800)
    if not context or not question or not answer:
        return None
    return {"id": row.get("id") or f"squad_{idx:06d}", "context": context, "question": question, "answer": answer}


def map_dolly(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    instruction = clipped(row.get("instruction") or "", 2400)
    context = clipped(row.get("context") or "", 6000)
    response = clipped(row.get("response") or "", 6000)
    if not instruction or not response:
        return None
    return {"id": f"dolly_{idx:05d}", "instruction": instruction, "context": context, "response": response}


def map_gsm8k(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    question = clipped(row.get("question") or "", 2400)
    answer = clipped(row.get("answer") or "", 6000)
    if not question or not answer:
        return None
    return {"id": f"gsm8k_{idx:05d}", "question": question, "answer": answer}


def map_tinystories(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    text = clipped(row.get("text") or "", 7000)
    if len(text.split()) < 40:
        return None
    return {"id": f"tinystories_{idx:06d}", "text": text}


def map_wikitext(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    text = clipped(row.get("text") or "", 5000)
    if len(text.split()) < 50 or text.strip().startswith("="):
        return None
    return {"id": f"wikitext103_{idx:06d}", "text": text}


def map_cnn_dm(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    article = clipped(row.get("article") or "", 9000)
    highlights = clipped(row.get("highlights") or "", 4000)
    if not article or not highlights:
        return None
    return {"id": f"cnn_dm_{idx:06d}", "article": article, "highlights": highlights}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--sources", default="squad,dolly,gsm8k,tinystories,wikitext103,cnn_dailymail")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--squad-limit", type=int, default=SOURCE_DEFAULTS["squad"])
    parser.add_argument("--dolly-limit", type=int, default=SOURCE_DEFAULTS["dolly"])
    parser.add_argument("--gsm8k-limit", type=int, default=SOURCE_DEFAULTS["gsm8k"])
    parser.add_argument("--tinystories-limit", type=int, default=SOURCE_DEFAULTS["tinystories"])
    parser.add_argument("--wikitext-limit", type=int, default=SOURCE_DEFAULTS["wikitext103"])
    parser.add_argument("--cnn-dm-limit", type=int, default=SOURCE_DEFAULTS["cnn_dailymail"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = {item.strip() for item in args.sources.split(",") if item.strip()}
    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Local Hugging Face raw cache for TMCRA token graph LM 500k seed building.",
        "streaming": bool(args.streaming),
        "sources": [],
        "benchmark_exclusion_note": "This cache contains only open Hugging Face datasets, not LongMemEval or internal benchmark outputs.",
    }

    specs = {
        "squad": ("rajpurkar/squad", None, "train", args.squad_limit, map_squad),
        "dolly": ("databricks/databricks-dolly-15k", None, "train", args.dolly_limit, map_dolly),
        "gsm8k": ("openai/gsm8k", "main", "train", args.gsm8k_limit, map_gsm8k),
        "tinystories": ("roneneldan/TinyStories", None, "train", args.tinystories_limit, map_tinystories),
        "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", args.wikitext_limit, map_wikitext),
        "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0", "train", args.cnn_dm_limit, map_cnn_dm),
    }

    write_progress(args.out_dir, {"status": "running", "source": None, "source_count": 0})
    for source, (dataset_name, config, split, limit, mapper) in specs.items():
        if source not in selected:
            continue
        info = dump_source(
            out_dir=args.out_dir,
            source=source,
            dataset_name=dataset_name,
            config=config,
            split=split,
            limit=limit,
            mapper=mapper,
            streaming=args.streaming,
            progress_every=args.progress_every,
        )
        manifest["sources"].append(info)
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(int(item["count"]) for item in manifest["sources"])
    manifest["total_records"] = total
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress(args.out_dir, {"status": "completed", "source": None, "source_count": 0, "total_records": total})
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
