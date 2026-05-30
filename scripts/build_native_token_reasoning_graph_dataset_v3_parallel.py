from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import shutil
import time
from pathlib import Path
from typing import Any

from build_native_token_reasoning_graph_dataset_v3 import (
    EDGE_TYPES,
    NODE_TYPES,
    _segments_for_annotation,
    build_graph,
)
from native_token_graph_common import LearnedBpeTokenizer


def count_rows(path: Path, *, limit: int) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                if limit and count >= limit:
                    break
    return count


def build_val_indices(total_rows: int, *, seed: int, val_ratio: float) -> set[int]:
    rng = random.Random(seed)
    split_indices = list(range(total_rows))
    rng.shuffle(split_indices)
    val_count_target = max(1, int(total_rows * val_ratio)) if total_rows > 5 else min(1, total_rows)
    return set(split_indices[:val_count_target])


def worker_main(
    worker_id: int,
    *,
    args_payload: dict[str, Any],
    tokenizer_payload: dict[str, Any],
    val_indices: set[int],
    total_rows: int,
    parts_dir: Path,
) -> None:
    input_jsonl = Path(args_payload["input_jsonl"])
    workers = int(args_payload["workers"])
    limit = int(args_payload["limit"])
    progress_every = int(args_payload["progress_every"])
    tokenizer = LearnedBpeTokenizer.from_json(tokenizer_payload)

    train_path = parts_dir / f"train.part{worker_id:02d}.jsonl"
    val_path = parts_dir / f"val.part{worker_id:02d}.jsonl"
    ann_path = parts_dir / f"annotation.part{worker_id:02d}.jsonl"
    count_path = parts_dir / f"counts.part{worker_id:02d}.json"

    started = time.perf_counter()
    train_count = 0
    val_count = 0
    annotation_count = 0
    seen = 0

    with input_jsonl.open("r", encoding="utf-8") as src, train_path.open("w", encoding="utf-8") as train_f, val_path.open(
        "w", encoding="utf-8"
    ) as val_f, ann_path.open("w", encoding="utf-8") as ann_f:
        for row_index, line in enumerate(src):
            if limit and row_index >= limit:
                break
            if not line.strip():
                continue
            if row_index % workers != worker_id:
                continue

            row = json.loads(line)
            graph = build_graph(
                row,
                tokenizer,
                max_query_tokens=int(args_payload["max_query_tokens"]),
                max_context_tokens=int(args_payload["max_context_tokens"]),
                max_unit_tokens=int(args_payload["max_unit_tokens"]),
                max_segments=int(args_payload["max_segments"]),
                max_same_piece_edges=int(args_payload["max_same_piece_edges"]),
                max_overlap_edges=int(args_payload["max_overlap_edges"]),
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
                        "segments": _segments_for_annotation(
                            row,
                            max_segments=int(args_payload["max_segments"]),
                            max_segment_chars=800,
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            annotation_count += 1
            seen += 1
            if progress_every and (seen == 1 or seen % progress_every == 0):
                print(
                    f"[worker {worker_id:02d}] built local={seen} "
                    f"global~={min(total_rows, worker_id + (seen - 1) * workers + 1)}/{total_rows} "
                    f"train={train_count} val={val_count} elapsed={time.perf_counter() - started:.2f}s",
                    flush=True,
                )

    count_path.write_text(
        json.dumps(
            {
                "worker_id": worker_id,
                "train_count": train_count,
                "val_count": val_count,
                "annotation_count": annotation_count,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[worker {worker_id:02d}] done train={train_count} val={val_count} "
        f"ann={annotation_count} elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )


def concat_parts(parts: list[Path], output: Path, *, delete_after: bool = False) -> None:
    with output.open("w", encoding="utf-8") as out_f:
        for path in parts:
            with path.open("r", encoding="utf-8") as in_f:
                shutil.copyfileobj(in_f, out_f)
            if delete_after:
                path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-query-tokens", type=int, default=96)
    parser.add_argument("--max-context-tokens", type=int, default=80)
    parser.add_argument("--max-unit-tokens", type=int, default=48)
    parser.add_argument("--max-segments", type=int, default=16)
    parser.add_argument("--max-same-piece-edges", type=int, default=192)
    parser.add_argument("--max-overlap-edges", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out_dir / "_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parallel] counting rows from {args.input_jsonl}", flush=True)
    total_rows = count_rows(args.input_jsonl, limit=args.limit)
    print(f"[parallel] total_rows={total_rows} workers={args.workers}", flush=True)
    val_indices = build_val_indices(total_rows, seed=args.seed, val_ratio=args.val_ratio)

    tokenizer_payload = json.loads(args.tokenizer_json.read_text(encoding="utf-8"))
    (args.out_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args_payload = {
        "input_jsonl": str(args.input_jsonl),
        "limit": args.limit,
        "workers": args.workers,
        "max_query_tokens": args.max_query_tokens,
        "max_context_tokens": args.max_context_tokens,
        "max_unit_tokens": args.max_unit_tokens,
        "max_segments": args.max_segments,
        "max_same_piece_edges": args.max_same_piece_edges,
        "max_overlap_edges": args.max_overlap_edges,
        "progress_every": args.progress_every,
    }
    processes: list[mp.Process] = []
    for worker_id in range(args.workers):
        proc = mp.Process(
            target=worker_main,
            args=(worker_id,),
            kwargs={
                "args_payload": args_payload,
                "tokenizer_payload": tokenizer_payload,
                "val_indices": val_indices,
                "total_rows": total_rows,
                "parts_dir": parts_dir,
            },
        )
        proc.start()
        processes.append(proc)

    failures: list[tuple[int, int | None]] = []
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            failures.append((proc.pid or -1, proc.exitcode))
    if failures:
        raise RuntimeError(f"worker failures: {failures}")

    print("[parallel] merging parts", flush=True)
    train_parts = [parts_dir / f"train.part{i:02d}.jsonl" for i in range(args.workers)]
    val_parts = [parts_dir / f"val.part{i:02d}.jsonl" for i in range(args.workers)]
    ann_parts = [parts_dir / f"annotation.part{i:02d}.jsonl" for i in range(args.workers)]
    concat_parts(train_parts, args.out_dir / "train.base.jsonl", delete_after=True)
    concat_parts(val_parts, args.out_dir / "val.base.jsonl", delete_after=True)
    concat_parts(ann_parts, args.out_dir / "annotation_input.jsonl", delete_after=True)

    train_count = 0
    val_count = 0
    annotation_count = 0
    worker_counts: list[dict[str, Any]] = []
    for worker_id in range(args.workers):
        payload = json.loads((parts_dir / f"counts.part{worker_id:02d}.json").read_text(encoding="utf-8"))
        worker_counts.append(payload)
        train_count += int(payload["train_count"])
        val_count += int(payload["val_count"])
        annotation_count += int(payload["annotation_count"])

    manifest = {
        "dataset_version": "native_token_reasoning_graph_v3_base",
        "input_jsonl": str(args.input_jsonl),
        "train_count": train_count,
        "val_count": val_count,
        "annotation_count": annotation_count,
        "vocab_size": len(tokenizer_payload.get("vocab", {})),
        "node_type_vocab": NODE_TYPES,
        "edge_type_vocab": EDGE_TYPES,
        "requires_teacher_annotation": True,
        "schema_note": "Reads token_graph_corpus_v2 fields first: source_segments, text_units, target_text.",
        "parallel_builder": True,
        "workers": args.workers,
        "tokenizer_json": str(args.tokenizer_json),
        "worker_counts": worker_counts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
