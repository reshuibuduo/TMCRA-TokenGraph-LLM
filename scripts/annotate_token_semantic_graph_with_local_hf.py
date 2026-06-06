from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from annotate_token_semantic_graph_with_openai import (
    SYSTEM_PROMPT,
    _safe_json_from_text,
    _validate_annotation,
    load_jsonl_limited,
    make_user_prompt,
)


def _load_model(
    model_name_or_path: str,
    *,
    load_in_4bit: bool,
    dtype_name: str,
    max_memory: str,
    cpu_max_memory: str,
    offload_folder: Path | None,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(str(dtype_name).lower(), torch.bfloat16)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    max_memory_map: dict[Any, str] = {}
    if max_memory and torch.cuda.is_available():
        max_memory_map[0] = max_memory
    if cpu_max_memory:
        max_memory_map["cpu"] = cpu_max_memory
    if max_memory_map:
        kwargs["max_memory"] = max_memory_map
    if offload_folder is not None:
        offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(offload_folder)
        kwargs["offload_state_dict"] = True
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    return tokenizer, model


def _chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"system: {system_prompt}\nuser: {user_prompt}\nassistant:"


@torch.inference_mode()
def generate_batch(
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    do_sample = temperature > 0
    outputs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(0.01, temperature) if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    texts: list[str] = []
    input_len = enc["input_ids"].shape[1]
    for row in outputs:
        texts.append(tokenizer.decode(row[input_len:], skip_special_tokens=True).strip())
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--progress-json", required=True, type=Path)
    parser.add_argument("--failure-jsonl", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-memory", default="")
    parser.add_argument("--cpu-max-memory", default="")
    parser.add_argument("--offload-folder", type=Path)
    parser.add_argument("--max-segments", type=int, default=12)
    parser.add_argument("--max-segment-chars", type=int, default=900)
    parser.add_argument("--max-target-chars", type=int, default=700)
    args = parser.parse_args()

    started = time.perf_counter()
    rows = load_jsonl_limited(args.input_jsonl, limit=args.limit + args.offset if args.limit else 0)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]

    done_ids: set[str] = set()
    if args.out_jsonl.exists():
        for item in load_jsonl_limited(args.out_jsonl):
            if item.get("schema_version") == "token_graph_corpus_v2" and item.get("semantic_teacher"):
                done_ids.add(str(item.get("sample_id", "")))
    pending = [row for row in rows if str(row.get("sample_id", "")) not in done_ids]

    tokenizer, model = _load_model(
        args.model,
        load_in_4bit=args.load_in_4bit,
        dtype_name=args.dtype,
        max_memory=args.max_memory,
        cpu_max_memory=args.cpu_max_memory,
        offload_folder=args.offload_folder,
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.progress_json.parent.mkdir(parents=True, exist_ok=True)
    failure_out = None
    if args.failure_jsonl is not None:
        args.failure_jsonl.parent.mkdir(parents=True, exist_ok=True)
        failure_out = args.failure_jsonl.open("a", encoding="utf-8")
    ok = len(done_ids)
    errors = 0
    try:
        with args.out_jsonl.open("a", encoding="utf-8") as out:
            for start in range(0, len(pending), max(1, args.batch_size)):
                batch_rows = pending[start : start + max(1, args.batch_size)]
                prompts = [
                    _chat_prompt(
                        tokenizer,
                        SYSTEM_PROMPT,
                        make_user_prompt(
                            row,
                            max_segments=args.max_segments,
                            max_segment_chars=args.max_segment_chars,
                            max_target_chars=args.max_target_chars,
                        ),
                    )
                    for row in batch_rows
                ]
                try:
                    texts = generate_batch(
                        tokenizer,
                        model,
                        prompts,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                except Exception as exc:
                    texts = []
                    errors += len(batch_rows)
                    last_error = repr(exc)
                    if failure_out is not None:
                        for row in batch_rows:
                            failure_out.write(
                                json.dumps(
                                    {
                                        "sample_id": str(row.get("sample_id", "")),
                                        "error_type": type(exc).__name__,
                                        "error": repr(exc),
                                        "raw_output": "",
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        failure_out.flush()
                else:
                    last_error = ""

                for row, text in zip(batch_rows, texts):
                    try:
                        annotation = _safe_json_from_text(text)
                        spans, edges = _validate_annotation(annotation, row)
                        enriched = dict(row)
                        enriched["semantic_spans"] = spans
                        enriched["semantic_edges"] = edges
                        enriched["semantic_teacher"] = {
                            "model": args.model,
                            "schema": "token_graph_semantic_v1",
                            "runtime": "local_hf",
                            "validated_span_count": len(spans),
                            "validated_edge_count": len(edges),
                        }
                        out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                        out.flush()
                        ok += 1
                    except Exception as exc:
                        errors += 1
                        last_error = repr(exc)
                        if failure_out is not None:
                            failure_out.write(
                                json.dumps(
                                    {
                                        "sample_id": str(row.get("sample_id", "")),
                                        "error_type": type(exc).__name__,
                                        "error": repr(exc),
                                        "raw_output": str(text),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            failure_out.flush()
                progress = {
                    "status": "running",
                    "total": len(rows),
                    "pending_initial": len(pending),
                    "ok_total": ok,
                    "error_count": errors,
                    "remaining_estimate": max(0, len(pending) - (start + len(batch_rows))),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "last_error": last_error[:240],
                }
                args.progress_json.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(progress, ensure_ascii=False), flush=True)
    finally:
        if failure_out is not None:
            failure_out.close()
    progress = {
        "status": "completed",
        "total": len(rows),
        "ok_total": ok,
        "error_count": errors,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": args.model,
        "batch_size": args.batch_size,
    }
    args.progress_json.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(progress, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
