from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from native_token_graph_common import load_jsonl


SYSTEM_PROMPT = """You are a training-time teacher for a graph-native causal language model.
Return strict JSON only. Do not include markdown.

Goal:
Extract token-level semantic graph supervision from the provided prompt/source text.
Do not answer the task. Do not create new facts. Use exact quotes from source_segments or text_units.

The output will be mapped to token nodes. Keep spans short, meaningful, and quote-exact.
Prefer relations that help a graph model learn language reasoning:
entity-attribute, cause-effect, condition-result, definition, example, temporal order,
quantity/count, coreference, contrast, support/evidence, and long-range tunnel links.
"""


ALLOWED_EDGE_TYPES = {
    "same_entity",
    "entity_attribute",
    "relation",
    "cause_effect",
    "condition_result",
    "temporal",
    "definition",
    "example",
    "contrast",
    "part_whole",
    "quantity",
    "coreference",
    "support",
    "negative",
    "tunnel",
}


def load_jsonl_limited(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _safe_json_from_text(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("teacher output is not an object")
    return payload


def _segments_for_prompt(row: dict[str, Any], *, max_segments: int, max_chars: int) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for idx, seg in enumerate(row.get("source_segments", []) or []):
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "segment_id": str(seg.get("segment_id", "") or f"s{idx + 1}"),
                "kind": str(seg.get("segment_type", "") or "source_segment"),
                "text": text[:max_chars],
            }
        )
        if len(segments) >= max_segments:
            return segments
    for idx, unit in enumerate(row.get("text_units", []) or []):
        text = str(unit.get("text", "") or unit.get("content", "") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "segment_id": str(unit.get("unit_id", "") or f"u{idx + 1}"),
                "kind": str(unit.get("unit_type", "") or "text_unit"),
                "text": text[:max_chars],
            }
        )
        if len(segments) >= max_segments:
            break
    return segments


def make_user_prompt(row: dict[str, Any], *, max_segments: int, max_segment_chars: int, max_target_chars: int) -> str:
    output_schema = {
        "sample_id": str(row.get("sample_id", "")),
        "semantic_spans": [
            {
                "span_id": "s1",
                "segment_id": "exact segment_id from input",
                "quote": "short exact quote copied from that segment",
                "role": "entity|attribute|action|cause|effect|condition|result|definition|example|temporal|quantity|support|negative",
                "confidence": 0.0,
            }
        ],
        "semantic_edges": [
            {
                "src_span_id": "s1",
                "dst_span_id": "s2",
                "edge_type": "same_entity|entity_attribute|relation|cause_effect|condition_result|temporal|definition|example|contrast|part_whole|quantity|coreference|support|negative|tunnel",
                "label": 1,
                "confidence": 0.0,
            }
        ],
    }
    payload = {
        "instruction": (
            "Produce token-level semantic graph supervision. "
            "Keep at most 10 semantic_spans and 14 semantic_edges. "
            "Every quote must be copied from the provided source text, not from target_text. "
            "Use target_text only to understand what kind of generation the graph should support. "
            "If source text is only an instruction, extract task type, constraints, quantities, requested format, and intent."
        ),
        "output_schema": output_schema,
        "sample": {
            "sample_id": str(row.get("sample_id", "")),
            "source": str(row.get("source", "")),
            "query": str(row.get("query", ""))[:max_segment_chars],
            "target_text_preview": str(row.get("target_text", "") or row.get("legacy", {}).get("answer", ""))[:max_target_chars],
            "segments": _segments_for_prompt(row, max_segments=max_segments, max_chars=max_segment_chars),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def call_openai_compat(
    *,
    base_url: str,
    model: str,
    api_key: str,
    row: dict[str, Any],
    timeout_s: int,
    temperature: float,
    max_tokens: int,
    max_segments: int,
    max_segment_chars: int,
    max_target_chars: int,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("missing API key")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": make_user_prompt(
                    row,
                    max_segments=max_segments,
                    max_segment_chars=max_segment_chars,
                    max_target_chars=max_target_chars,
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8", errors="replace")
    body = json.loads(raw)
    content = body["choices"][0]["message"]["content"]
    parsed = _safe_json_from_text(content)
    parsed.setdefault("sample_id", str(row.get("sample_id", "")))
    return parsed


def _validate_annotation(annotation: dict[str, Any], row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_segment_ids = {
        str(seg.get("segment_id", ""))
        for seg in (row.get("source_segments", []) or [])
        if str(seg.get("segment_id", ""))
    }
    valid_segment_ids.update(
        str(unit.get("unit_id", ""))
        for unit in (row.get("text_units", []) or [])
        if str(unit.get("unit_id", ""))
    )
    spans: list[dict[str, Any]] = []
    span_ids: set[str] = set()
    for idx, span in enumerate(annotation.get("semantic_spans", []) or []):
        if not isinstance(span, dict):
            continue
        quote = str(span.get("quote", "") or span.get("text", "") or "").strip()
        segment_id = str(span.get("segment_id", "") or span.get("source_segment_id", "") or span.get("unit_id", "")).strip()
        if not quote or not segment_id or segment_id not in valid_segment_ids:
            continue
        span_id = str(span.get("span_id", "") or span.get("id", "") or f"sem_{idx:03d}").strip()
        if not span_id or span_id in span_ids:
            span_id = f"sem_{idx:03d}"
        span_ids.add(span_id)
        spans.append(
            {
                "span_id": span_id,
                "segment_id": segment_id,
                "quote": quote[:240],
                "role": str(span.get("role", "") or "semantic"),
                "confidence": float(span.get("confidence", 0.7) or 0.0),
            }
        )
        if len(spans) >= 10:
            break
    edges: list[dict[str, Any]] = []
    for edge in annotation.get("semantic_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src_span_id", "") or edge.get("src", "") or edge.get("source", "")).strip()
        dst = str(edge.get("dst_span_id", "") or edge.get("dst", "") or edge.get("target", "")).strip()
        edge_type = str(edge.get("edge_type", "") or edge.get("relation", "") or "relation").strip().lower().replace("-", "_").replace(" ", "_")
        if src not in span_ids or dst not in span_ids or src == dst:
            continue
        if edge_type not in ALLOWED_EDGE_TYPES:
            edge_type = "relation"
        edges.append(
            {
                "src_span_id": src,
                "dst_span_id": dst,
                "edge_type": edge_type,
                "label": int(edge.get("label", 1) or 0),
                "confidence": float(edge.get("confidence", 0.7) or 0.0),
            }
        )
        if len(edges) >= 14:
            break
    return spans, edges


def annotate_one(args: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    index, row, cfg = args
    last_error = ""
    for attempt in range(int(cfg["retries"]) + 1):
        try:
            annotation = call_openai_compat(
                base_url=str(cfg["base_url"]),
                model=str(cfg["model"]),
                api_key=str(cfg["api_key"]),
                row=row,
                timeout_s=int(cfg["timeout_s"]),
                temperature=float(cfg["temperature"]),
                max_tokens=int(cfg["max_tokens"]),
                max_segments=int(cfg["max_segments"]),
                max_segment_chars=int(cfg["max_segment_chars"]),
                max_target_chars=int(cfg["max_target_chars"]),
            )
            spans, edges = _validate_annotation(annotation, row)
            enriched = dict(row)
            enriched["semantic_spans"] = spans
            enriched["semantic_edges"] = edges
            enriched["semantic_teacher"] = {
                "model": str(cfg["model"]),
                "base_url_configured": bool(cfg["base_url"]),
                "schema": "token_graph_semantic_v1",
                "raw_span_count": len(annotation.get("semantic_spans", []) or []),
                "raw_edge_count": len(annotation.get("semantic_edges", []) or []),
                "validated_span_count": len(spans),
                "validated_edge_count": len(edges),
            }
            return {"index": index, "sample_id": row.get("sample_id", ""), "status": "ok", "row": enriched, "error": ""}
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(min(2.0 + attempt, 8.0))
    return {"index": index, "sample_id": row.get("sample_id", ""), "status": "error", "row": None, "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--progress-json", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--api-key-env", default="TOKEN_SEMANTIC_TEACHER_API_KEY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-segments", type=int, default=12)
    parser.add_argument("--max-segment-chars", type=int, default=900)
    parser.add_argument("--max-target-chars", type=int, default=700)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    rows = load_jsonl_limited(args.input_jsonl, limit=args.limit)
    done_ids: set[str] = set()
    if args.out_jsonl.exists():
        for item in load_jsonl(args.out_jsonl):
            if item.get("schema_version") == "token_graph_corpus_v2" and item.get("semantic_teacher"):
                done_ids.add(str(item.get("sample_id", "")))
    pending = [(idx, row) for idx, row in enumerate(rows) if str(row.get("sample_id", "")) not in done_ids]
    cfg = {
        "base_url": args.base_url,
        "model": args.model,
        "api_key": api_key,
        "timeout_s": args.timeout_s,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_segments": args.max_segments,
        "max_segment_chars": args.max_segment_chars,
        "max_target_chars": args.max_target_chars,
        "retries": args.retries,
    }
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.progress_json.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    counts = {"ok": len(done_ids), "error": 0}
    with args.out_jsonl.open("a", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(annotate_one, (idx, row, cfg)) for idx, row in pending]
            for completed, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                item = fut.result()
                counts[item["status"]] = counts.get(item["status"], 0) + 1
                if item["status"] == "ok" and item["row"] is not None:
                    out.write(json.dumps(item["row"], ensure_ascii=False) + "\n")
                    out.flush()
                progress = {
                    "status": "running",
                    "input_jsonl": str(args.input_jsonl),
                    "out_jsonl": str(args.out_jsonl),
                    "total": len(rows),
                    "already_done": len(done_ids),
                    "pending_initial": len(pending),
                    "completed_this_run": completed,
                    "ok_total_estimate": counts.get("ok", 0),
                    "error_count": counts.get("error", 0),
                    "remaining_estimate": max(0, len(pending) - completed),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "model": args.model,
                    "base_url_configured": bool(args.base_url),
                    "api_key_configured": bool(api_key),
                    "last_error": item["error"][:240],
                }
                args.progress_json.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(progress, ensure_ascii=False), flush=True)
    progress = {
        "status": "completed",
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "total": len(rows),
        "ok_total": counts.get("ok", 0),
        "error_count": counts.get("error", 0),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": args.model,
        "base_url_configured": bool(args.base_url),
        "api_key_configured": bool(api_key),
    }
    args.progress_json.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(progress, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
