#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_gzip(src: Path) -> tuple[bool, int]:
    dst = src.with_suffix(src.suffix + ".gz")
    if dst.exists():
        return False, dst.stat().st_size
    with src.open("rb") as fin, gzip.open(dst, "wb", compresslevel=9) as fout:
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    return True, dst.stat().st_size


def main() -> None:
    candidates = sorted(list(DATA.glob("*.jsonl")) + list(DATA.glob("*.csv")))
    compressed = []
    totals = {"input_bytes": 0, "gz_bytes": 0, "savings_bytes": 0}

    for f in candidates:
        created, gz_size = ensure_gzip(f)
        in_size = f.stat().st_size
        totals["input_bytes"] += in_size
        totals["gz_bytes"] += gz_size
        totals["savings_bytes"] += max(in_size - gz_size, 0)
        compressed.append(
            {
                "file": str(f.relative_to(ROOT)),
                "size_bytes": in_size,
                "gzip": {
                    "file": str((f.with_suffix(f.suffix + ".gz")).relative_to(ROOT)),
                    "size_bytes": gz_size,
                    "created_now": created,
                    "sha256": sha256_file(f.with_suffix(f.suffix + ".gz")),
                },
                "compression_ratio": round(gz_size / in_size, 4) if in_size else 0,
            }
        )

    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "timestamp": ts,
        "step": "storage_runtime_optimization_v2",
        "actions": [
            "ensured gzip mirrors for all jsonl/csv artifacts",
            "computed integrity hashes for compressed artifacts",
            "generated reclaim estimate if switching readers to .gz inputs",
        ],
        "totals": totals,
        "estimated_reclaim_mb": round(totals["savings_bytes"] / (1024 * 1024), 3),
        "files": compressed,
        "next_runtime_step": "teach training/report scripts to auto-read .gz to avoid uncompressed duplication",
    }

    out = REPORTS / f"storage_runtime_optimization_v2_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S%z')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
