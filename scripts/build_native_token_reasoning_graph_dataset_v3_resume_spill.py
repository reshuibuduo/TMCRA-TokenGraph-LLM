from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
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


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


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
    resume_local_count: int,
    spill_parts_dir: Path,
) -> None:
    input_jsonl = Path(args_payload["input_jsonl"])
    workers = int(args_payload["workers"])
    limit = int(args_payload["limit"])
    progress_every = int(args_payload["progress_every"])
    tokenizer = LearnedBpeTokenizer.from_json(tokenizer_payload)

    train_path = spill_parts_dir / f"train.resume_part{worker_id:02d}.jsonl"
    val_path = spill_parts_dir / f"val.resume_part{worker_id:02d}.jsonl"
    ann_path = spill_parts_dir / f"annotation.resume_part{worker_id:02d}.jsonl"
    count_path = spill_parts_dir / f"counts.resume_part{worker_id:02d}.json"

    started = time.perf_counter()
    train_count = 0
    val_count = 0
    annotation_count = 0
    assigned_seen = 0
    resumed_from = resume_local_count

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

            if assigned_seen < resumed_from:
                assigned_seen += 1
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
            assigned_seen += 1
            if progress_every and (annotation_count == 1 or annotation_count % progress_every == 0):
                print(
                    f"[resume worker {worker_id:02d}] new={annotation_count} "
                    f"assigned_seen={assigned_seen} global~={min(total_rows, worker_id + (assigned_seen - 1) * workers + 1)}/{total_rows} "
                    f"train={train_count} val={val_count} elapsed={time.perf_counter() - started:.2f}s",
                    flush=True,
                )

    count_path.write_text(
        json.dumps(
            {
                "worker_id": worker_id,
                "resume_local_count": resumed_from,
                "train_count": train_count,
                "val_count": val_count,
                "annotation_count": annotation_count,
                "final_assigned_seen": assigned_seen,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[resume worker {worker_id:02d}] done new_train={train_count} new_val={val_count} "
        f"new_ann={annotation_count} final_assigned_seen={assigned_seen} elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )


def rel_or_abs(path: Path) -> str:
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--spill-dir", required=True, type=Path)
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
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    started = time.perf_counter()
    base_parts_dir = args.out_dir / "_parts"
    if not base_parts_dir.exists():
        raise FileNotFoundError(f"Missing existing base parts dir: {base_parts_dir}")
    args.spill_dir.mkdir(parents=True, exist_ok=True)
    spill_parts_dir = args.spill_dir / "_resume_parts"
    spill_parts_dir.mkdir(parents=True, exist_ok=True)

    total_rows = count_rows(args.input_jsonl, limit=args.limit)
    val_indices = build_val_indices(total_rows, seed=args.seed, val_ratio=args.val_ratio)
    tokenizer_payload = json.loads(args.tokenizer_json.read_text(encoding="utf-8"))
    (args.out_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resume_counts: list[int] = []
    for worker_id in range(args.workers):
        resume_counts.append(count_jsonl_lines(base_parts_dir / f"annotation.part{worker_id:02d}.jsonl"))
    print(
        json.dumps(
            {
                "mode": "resume_spill",
                "total_rows": total_rows,
                "workers": args.workers,
                "resume_total": sum(resume_counts),
                "resume_min": min(resume_counts) if resume_counts else 0,
                "resume_max": max(resume_counts) if resume_counts else 0,
                "spill_parts_dir": str(spill_parts_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
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
                "resume_local_count": resume_counts[worker_id],
                "spill_parts_dir": spill_parts_dir,
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
        raise RuntimeError(f"resume worker failures: {failures}")

    train_shards = [base_parts_dir / f"train.part{i:02d}.jsonl" for i in range(args.workers)] + [
        spill_parts_dir / f"train.resume_part{i:02d}.jsonl" for i in range(args.workers)
    ]
    val_shards = [base_parts_dir / f"val.part{i:02d}.jsonl" for i in range(args.workers)] + [
        spill_parts_dir / f"val.resume_part{i:02d}.jsonl" for i in range(args.workers)
    ]
    annotation_shards = [base_parts_dir / f"annotation.part{i:02d}.jsonl" for i in range(args.workers)] + [
        spill_parts_dir / f"annotation.resume_part{i:02d}.jsonl" for i in range(args.workers)
    ]

    worker_counts: list[dict[str, Any]] = []
    train_count = 0
    val_count = 0
    annotation_count = 0
    for worker_id in range(args.workers):
        base_train = count_jsonl_lines(base_parts_dir / f"train.part{worker_id:02d}.jsonl")
        base_val = count_jsonl_lines(base_parts_dir / f"val.part{worker_id:02d}.jsonl")
        base_ann = count_jsonl_lines(base_parts_dir / f"annotation.part{worker_id:02d}.jsonl")
        resume_payload = json.loads((spill_parts_dir / f"counts.resume_part{worker_id:02d}.json").read_text(encoding="utf-8"))
        payload = {
            "worker_id": worker_id,
            "base_train_count": base_train,
            "base_val_count": base_val,
            "base_annotation_count": base_ann,
            **resume_payload,
        }
        worker_counts.append(payload)
        train_count += base_train + int(resume_payload["train_count"])
        val_count += base_val + int(resume_payload["val_count"])
        annotation_count += base_ann + int(resume_payload["annotation_count"])

    manifest = {
        "dataset_version": "native_token_reasoning_graph_v3_base_sharded_resume_spill",
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
        "sharded_dataset": True,
        "resume_spill": True,
        "workers": args.workers,
        "tokenizer_json": str(args.tokenizer_json),
        "train_shards": [rel_or_abs(path) for path in train_shards],
        "val_shards": [rel_or_abs(path) for path in val_shards],
        "annotation_shards": [rel_or_abs(path) for path in annotation_shards],
        "worker_counts": worker_counts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
