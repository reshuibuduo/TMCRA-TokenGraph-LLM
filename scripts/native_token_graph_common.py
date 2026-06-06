from __future__ import annotations

import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPACE = "\u2581"
DEFAULT_ALPHABET = (
    SPACE
    + "abcdefghijklmnopqrstuvwxyz"
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    + "0123456789"
    + ".,;:!?'-\"/\\()[]{}%$&+*=<>@#_|`~"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    return (SPACE + value.replace(" ", SPACE)) if value else ""


class LearnedBpeTokenizer:
    """Small learned char-BPE tokenizer for smoke training.

    This is a statistical tokenizer, not a hand-written domain rule system.
    It is intentionally simple so the experimental graph language model can
    run without external tokenizer dependencies.
    """

    def __init__(self, merges: list[tuple[str, str]] | None = None, vocab: dict[str, int] | None = None) -> None:
        self.merges = list(merges or [])
        self.vocab = dict(vocab or {})
        # Only apply merges whose resulting token is representable. Otherwise
        # encoding can create an out-of-vocabulary merged piece even when all
        # source characters are known.
        if self.vocab:
            self.merge_rank = {pair: idx for idx, pair in enumerate(self.merges) if pair[0] + pair[1] in self.vocab}
        else:
            self.merge_rank = {pair: idx for idx, pair in enumerate(self.merges)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self._encode_cache: OrderedDict[tuple[str, int | None], tuple[str, ...]] = OrderedDict()
        self.encode_cache_size = 50000

    @staticmethod
    def _symbols(text: str) -> list[str]:
        return list(normalize_text(text))

    @classmethod
    def train(
        cls,
        texts: list[str],
        *,
        vocab_size: int,
        min_pair_freq: int,
        max_text_chars: int,
        progress_every: int = 100,
    ) -> "LearnedBpeTokenizer":
        corpus = [cls._symbols(text[:max_text_chars]) for text in texts if str(text or "").strip()]
        merges: list[tuple[str, str]] = []
        base_vocab = {symbol for row in corpus for symbol in row}
        base_vocab.update(DEFAULT_ALPHABET)
        target_merges = max(0, int(vocab_size) - len(base_vocab) - 4)
        print(
            f"[bpe] corpus={len(corpus)} base_vocab={len(base_vocab)} target_merges={target_merges}",
            file=sys.stderr,
            flush=True,
        )
        for merge_idx in range(target_merges):
            pairs: Counter[tuple[str, str]] = Counter()
            for row in corpus:
                pairs.update(zip(row, row[1:]))
            if not pairs:
                break
            (left, right), freq = pairs.most_common(1)[0]
            if freq < min_pair_freq:
                break
            merges.append((left, right))
            if progress_every and (merge_idx + 1 == 1 or (merge_idx + 1) % progress_every == 0):
                print(f"[bpe] merge={merge_idx + 1} freq={freq}", file=sys.stderr, flush=True)
            merged = left + right
            next_corpus: list[list[str]] = []
            for row in corpus:
                out: list[str] = []
                i = 0
                while i < len(row):
                    if i + 1 < len(row) and row[i] == left and row[i + 1] == right:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(row[i])
                        i += 1
                next_corpus.append(out)
            corpus = next_corpus
        learned = Counter(symbol for row in corpus for symbol in row)
        ordered = [PAD, BOS, EOS, UNK]
        # Keep the base alphabet in the vocabulary. Without this, the final BPE
        # corpus can hide common characters inside merged pieces, and unseen
        # words later collapse to <unk>.
        for token in sorted(base_vocab, key=lambda item: (item != SPACE, item)):
            if token not in ordered:
                ordered.append(token)
        for token, _ in learned.most_common():
            if token not in ordered:
                ordered.append(token)
            if len(ordered) >= int(vocab_size):
                break
        return cls(merges=merges, vocab={token: idx for idx, token in enumerate(ordered)})

    def encode_pieces(self, text: str, *, max_tokens: int | None = None) -> list[str]:
        key = (str(text or ""), max_tokens)
        cached = self._encode_cache.get(key)
        if cached is not None:
            self._encode_cache.move_to_end(key)
            return list(cached)
        symbols = self._symbols(text)
        if not symbols:
            return []
        while True:
            best_rank = None
            best_pair = None
            for pair in zip(symbols, symbols[1:]):
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            left, right = best_pair
            merged = left + right
            out: list[str] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        if max_tokens is not None:
            symbols = symbols[:max_tokens]
        self._encode_cache[key] = tuple(symbols)
        if len(self._encode_cache) > self.encode_cache_size:
            self._encode_cache.popitem(last=False)
        return symbols

    def encode(self, text: str, *, max_tokens: int | None = None) -> list[int]:
        unk = self.vocab.get(UNK, 3)
        return [self.vocab.get(piece, unk) for piece in self.encode_pieces(text, max_tokens=max_tokens)]

    def decode(self, ids: list[int]) -> str:
        pieces = [self.id_to_token.get(int(idx), UNK) for idx in ids]
        text = "".join(piece for piece in pieces if piece not in {PAD, BOS, EOS})
        return text.replace(SPACE, " ").strip()

    def to_json(self) -> dict[str, Any]:
        return {"type": "learned_char_bpe_v2", "space_symbol": SPACE, "merges": self.merges, "vocab": self.vocab}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "LearnedBpeTokenizer":
        if payload.get("type") in {"huggingface_bpe_v1", "hf_pretrained_tokenizer_v1"}:
            return HuggingFaceBpeTokenizer.from_json(payload)
        if "model" in payload and ("pre_tokenizer" in payload or "decoder" in payload):
            return HuggingFaceBpeTokenizer(json.dumps(payload))
        return cls(merges=[tuple(pair) for pair in payload.get("merges", [])], vocab=dict(payload.get("vocab", {})))


class HuggingFaceBpeTokenizer:
    """Mature BPE tokenizer backed by Hugging Face tokenizers.

    This keeps tokenization statistical/learned while avoiding the brittle
    character-BPE behavior that produced many residual word fragments.
    """

    def __init__(self, tokenizer_json: str) -> None:
        try:
            from tokenizers import Tokenizer
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Install `tokenizers` to use HuggingFaceBpeTokenizer") from exc
        self._tokenizer = Tokenizer.from_str(tokenizer_json)
        self._tokenizer.add_special_tokens([PAD, BOS, EOS, UNK])
        self.tokenizer_json = self._tokenizer.to_str()
        self.vocab = dict(self._tokenizer.get_vocab())
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    @classmethod
    def train(
        cls,
        texts: list[str],
        *,
        vocab_size: int,
        min_frequency: int,
    ) -> "HuggingFaceBpeTokenizer":
        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.models import BPE
            from tokenizers.normalizers import Sequence, NFKC
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.trainers import BpeTrainer
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Install `tokenizers` to train HuggingFaceBpeTokenizer") from exc

        tokenizer = Tokenizer(BPE(unk_token=UNK))
        tokenizer.normalizer = Sequence([NFKC()])
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            special_tokens=[PAD, BOS, EOS, UNK],
            show_progress=True,
        )
        tokenizer.train_from_iterator((str(text or "") for text in texts if str(text or "").strip()), trainer=trainer)
        return cls(tokenizer.to_str())

    @classmethod
    def from_pretrained_model(cls, model_name: str) -> "HuggingFaceBpeTokenizer":
        """Load a mature LLM tokenizer and persist its backend tokenizer JSON.

        The graph language model still trains from scratch. This only reuses the
        tokenizer vocabulary and byte/subword boundary behavior from an existing
        LLM tokenizer, avoiding the tiny char-BPE fragmentation seen in long
        generation probes.
        """
        try:
            from tokenizers import Tokenizer
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Install `tokenizers` to load a pretrained tokenizer") from exc

        tokenizer_path = Path(str(model_name))
        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_pretrained(str(model_name))
        tokenizer.add_special_tokens([PAD, BOS, EOS, UNK])
        wrapped = cls(tokenizer.to_str())
        wrapped.pretrained_model_name = str(model_name)
        return wrapped

    def encode_pieces(self, text: str, *, max_tokens: int | None = None) -> list[str]:
        encoded = self._tokenizer.encode(str(text or ""))
        tokens = list(encoded.tokens)
        return tokens[:max_tokens] if max_tokens is not None else tokens

    def encode(self, text: str, *, max_tokens: int | None = None) -> list[int]:
        encoded = self._tokenizer.encode(str(text or ""))
        ids = list(encoded.ids)
        return ids[:max_tokens] if max_tokens is not None else ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode([int(idx) for idx in ids], skip_special_tokens=True).strip()

    def to_json(self) -> dict[str, Any]:
        payload = {"type": "huggingface_bpe_v1", "tokenizer_json": self.tokenizer_json, "vocab": self.vocab}
        model_name = getattr(self, "pretrained_model_name", None)
        if model_name:
            payload["type"] = "hf_pretrained_tokenizer_v1"
            payload["pretrained_model_name"] = model_name
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "HuggingFaceBpeTokenizer":
        return cls(str(payload["tokenizer_json"]))
