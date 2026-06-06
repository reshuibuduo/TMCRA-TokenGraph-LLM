#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a gzip schema2 file into robust transfer chunks.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=1)
    args = parser.parse_args()

    if args.chunk_mib < 1:
        raise SystemExit("--chunk-mib must be >= 1")

    src = args.input
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for old in out_dir.glob("part_*.bin"):
        old.unlink()

    chunk_size = args.chunk_mib * 1024 * 1024
    chunks: list[dict[str, object]] = []
    with src.open("rb") as f:
        index = 0
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            name = f"part_{index:05d}.bin"
            part = out_dir / name
            part.write_bytes(data)
            chunks.append(
                {
                    "name": name,
                    "index": index,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            index += 1

    manifest = {
        "source_file": str(src),
        "source_name": src.name,
        "source_size": src.stat().st_size,
        "source_sha256": sha256_file(src),
        "chunk_mib": args.chunk_mib,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("source_name", "source_size", "chunk_count", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
