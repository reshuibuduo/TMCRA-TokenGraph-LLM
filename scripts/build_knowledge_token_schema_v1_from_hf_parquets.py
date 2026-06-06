from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


PARQUET_SPECS = [
    {
        "key": "squad",
        "repo": "squad",
        "file": "plain_text/train-00000-of-00001.parquet",
        "quota": 500,
    },
    {
        "key": "sciq",
        "repo": "sciq",
        "file": "data/train-00000-of-00001.parquet",
        "quota": 500,
    },
]


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def words(text: str, *, limit: int) -> list[str]:
    out = [m.group(0) for m in WORD_RE.finditer(text or "")]
    return out[:limit]


def normalize_token_text(text: str) -> str:
    return normalize_space(text).lower()


def split_sentences(text: str, *, limit: int) -> list[str]:
    parts = [part.strip() for part in SENTENCE_RE.split(text or "") if part.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    return parts[:limit]


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


def add_bpe_items(
    *,
    tokenizer: Tokenizer,
    items: list[dict[str, str]],
    edges: list[dict[str, str]],
    prefix: str,
    text: str,
    role: str,
    limit: int,
    support: set[str] | None = None,
    negative: set[str] | None = None,
) -> list[str]:
    support = support if support is not None else set()
    negative = negative if negative is not None else set()
    ids: list[str] = []
    prev_id: str | None = None
    encoded = tokenizer.encode(str(text or ""))
    token_pairs = list(zip(encoded.tokens, encoded.ids))[:limit]
    for pos, (piece, piece_id) in enumerate(token_pairs):
        token_id = f"{prefix}:{pos:04d}"
        ids.append(token_id)
        token_text = tokenizer.decode([int(piece_id)]).strip() or str(piece)
        items.append(
            {
                "token_id": token_id,
                "text": token_text,
                "piece": str(piece),
                "piece_id": int(piece_id),
                "role": role,
            }
        )
        if prev_id is not None:
            edges.append({"src": prev_id, "dst": token_id, "edge_type": "knowledge_sequence"})
        prev_id = token_id
    return ids


def add_query_links(query_ids: list[str], knowledge_ids: list[str], edges: list[dict[str, str]]) -> None:
    query_terms = {qid.split(":", 1)[-1]: qid for qid in query_ids}
    # Link by lexical overlap through item text outside this helper; kept simple
    # to avoid creating a rule-based runtime path. This only creates supervised
    # graph structure in the training data.
    for qid in query_ids[:24]:
        for kid in knowledge_ids[:96]:
            if qid == kid:
                continue
            if qid.rsplit(":", 1)[-1] == kid.rsplit(":", 1)[-1]:
                edges.append({"src": qid, "dst": kid, "edge_type": "query_to_knowledge"})
                break


def build_squad_record(row: dict[str, Any], *, tokenizer: Tokenizer, sample_id: str, split: str, source: str) -> dict[str, Any] | None:
    question = normalize_space(row.get("question", ""))
    context = normalize_space(row.get("context", ""))
    answers = row.get("answers") or {}
    answer_texts = answers.get("text") if isinstance(answers, dict) else None
    answer = normalize_space(answer_texts[0] if answer_texts else "")
    if not question or not context or not answer:
        return None
    answer_sentence = ""
    for sentence in split_sentences(context, limit=48):
        if answer.lower() in sentence.lower():
            answer_sentence = sentence
            break
    if not answer_sentence:
        answer_sentence = " ".join(words(context, limit=80))
    return make_knowledge_record(
        tokenizer=tokenizer,
        sample_id=sample_id,
        source=source,
        task_family="squad",
        query=f"Question: {question}\nAnswer in one short sentence from the knowledge graph.",
        target_text=f"The answer is {answer}.",
        split=split,
        support_text=answer_sentence,
        answer_text=answer,
        choices=[],
        source_dataset="squad",
    )


def build_sciq_record(row: dict[str, Any], *, tokenizer: Tokenizer, sample_id: str, split: str, source: str) -> dict[str, Any] | None:
    question = normalize_space(row.get("question", ""))
    answer = normalize_space(row.get("correct_answer", ""))
    support = normalize_space(row.get("support", ""))
    distractors = [normalize_space(row.get(f"distractor{i}", "")) for i in range(1, 4)]
    if not question or not answer:
        return None
    return make_knowledge_record(
        tokenizer=tokenizer,
        sample_id=sample_id,
        source=source,
        task_family="sciq",
        query=f"Question: {question}\nAnswer briefly from the token graph.",
        target_text=f"The answer is {answer}.",
        split=split,
        support_text=support or question,
        answer_text=answer,
        choices=[answer] + [item for item in distractors if item],
        source_dataset="sciq",
    )


def make_knowledge_record(
    *,
    tokenizer: Tokenizer,
    sample_id: str,
    source: str,
    task_family: str,
    query: str,
    target_text: str,
    split: str,
    support_text: str,
    answer_text: str,
    choices: list[str],
    source_dataset: str,
) -> dict[str, Any]:
    items: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    support_ids: set[str] = set()
    negative_ids: set[str] = set()

    query_ids = add_bpe_items(tokenizer=tokenizer, items=items, edges=edges, prefix="q", text=query, role="query_term", limit=96)
    support_token_ids = add_bpe_items(
        tokenizer=tokenizer,
        items=items,
        edges=edges,
        prefix="k",
        text=support_text,
        role="support_entity",
        limit=160,
        support=support_ids,
    )
    answer_token_ids = add_bpe_items(
        tokenizer=tokenizer,
        items=items,
        edges=edges,
        prefix="a",
        text=answer_text,
        role="answer_value",
        limit=48,
        support=support_ids,
    )
    support_ids.update(answer_token_ids)
    answer_piece_ids = {str(item.get("piece_id")) for item in items if item.get("token_id") in set(answer_token_ids)}
    for sid in support_token_ids:
        piece_id = str(next((item["piece_id"] for item in items if item["token_id"] == sid), ""))
        if piece_id in answer_piece_ids:
            support_ids.add(sid)
            for aid in answer_token_ids:
                edges.append({"src": sid, "dst": aid, "edge_type": "support_path"})

    for choice_index, choice in enumerate(choices[:8]):
        role = "answer_choice" if choice.strip().lower() == answer_text.strip().lower() else "negative_choice"
        choice_ids = add_bpe_items(
            tokenizer=tokenizer,
            items=items,
            edges=edges,
            prefix=f"c{choice_index}",
            text=choice,
            role=role,
            limit=48,
        )
        if role == "answer_choice":
            support_ids.update(choice_ids)
            for cid in choice_ids:
                for aid in answer_token_ids:
                    edges.append({"src": cid, "dst": aid, "edge_type": "support_path"})
        else:
            negative_ids.update(choice_ids)
            for cid in choice_ids:
                for aid in answer_token_ids[:4]:
                    edges.append({"src": cid, "dst": aid, "edge_type": "negative_path"})

    token_text_by_id = {item["token_id"]: normalize_token_text(item["text"]) for item in items}
    for qid in query_ids[:48]:
        qtext = token_text_by_id.get(qid, "")
        if len(qtext) <= 2:
            continue
        linked = 0
        for kid in support_token_ids + answer_token_ids:
            if token_text_by_id.get(kid, "") == qtext:
                edges.append({"src": qid, "dst": kid, "edge_type": "query_to_knowledge"})
                linked += 1
                if linked >= 4:
                    break

    return {
        "schema_version": "token_graph_knowledge_v1",
        "sample_id": sample_id,
        "source": source,
        "split": split,
        "task_family": task_family,
        "query": query,
        "question_date": None,
        "source_segments": [],
        "text_units": [],
        "knowledge_tokens": items,
        "knowledge_edges": edges,
        "support_segment_ids": [],
        "support_text_unit_ids": [],
        "support_knowledge_token_ids": sorted(support_ids),
        "negative_knowledge_token_ids": sorted(negative_ids),
        "target_text": target_text.strip(),
        "target_tokens": tokenizer.encode(target_text).tokens[:128],
        "corpus_note": "open_source_evidence_backed_knowledge_token_graph",
        "source_dataset": source_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-label", default="knowledge_token_v1")
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        default=Path(__file__).resolve().parent / "pretrained_tokenizers" / "gpt2" / "tokenizer.json",
    )
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--quotas", default="")
    parser.add_argument("--val-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    quotas = parse_quotas(args.quotas)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    errors: dict[str, str] = {}
    downloaded: dict[str, str] = {}
    kept = 0

    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for spec in PARQUET_SPECS:
            key = str(spec["key"])
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
                    split = "val" if rng.random() < args.val_ratio else "train"
                    source = f"{args.dataset_label}_{key}"
                    sample_id = f"{args.dataset_label}_{key}_{counts[key] + 1:07d}"
                    if key == "squad":
                        record = build_squad_record(row, tokenizer=tokenizer, sample_id=sample_id, split=split, source=source)
                    elif key == "sciq":
                        record = build_sciq_record(row, tokenizer=tokenizer, sample_id=sample_id, split=split, source=source)
                    else:
                        record = None
                    if record is None:
                        continue
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
        "tokenizer_json": str(args.tokenizer_json),
        "schema_version": "token_graph_knowledge_v1",
        "note": "Evidence-backed open-source knowledge datasets converted to token-level knowledge graph schema. Default sources are SQuAD answer-containing context sentences and SciQ support passages. Knowledge items are GPT-2 BPE tokens, not generated chunks.",
    }
    args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
