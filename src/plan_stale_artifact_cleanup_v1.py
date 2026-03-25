#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def bytes_to_mb(n: int) -> float:
    return round(n / (1024 * 1024), 4)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S%z")

    candidates = []
    total_bytes = 0

    for p in sorted(DATA_DIR.glob("*")):
        if not p.is_file():
            continue
        name = p.name
        # consider plain jsonl/csv only; if compressed twin exists, mark stale candidate
        if name.endswith(".jsonl"):
            gz = p.with_name(name + ".gz")
        elif name.endswith(".csv"):
            gz = p.with_name(name + ".gz")
        else:
            continue

        if gz.exists() and gz.is_file():
            size = p.stat().st_size
            total_bytes += size
            candidates.append(
                {
                    "path": str(p.relative_to(ROOT)),
                    "size_bytes": size,
                    "size_mb": bytes_to_mb(size),
                    "compressed_twin": str(gz.relative_to(ROOT)),
                    "compressed_size_bytes": gz.stat().st_size,
                    "compressed_size_mb": bytes_to_mb(gz.stat().st_size),
                }
            )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "step": "safe_stale_artifact_cleanup_planning",
        "dry_run": True,
        "candidate_count": len(candidates),
        "reclaimable_bytes_if_deleted": total_bytes,
        "reclaimable_mb_if_deleted": bytes_to_mb(total_bytes),
        "candidates": candidates,
        "note": "Plan-only report. No files were deleted.",
    }

    out = REPORTS_DIR / f"stale_artifact_cleanup_plan_v1_{ts}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "report": str(out.relative_to(ROOT)),
        "candidate_count": len(candidates),
        "reclaimable_mb_if_deleted": report["reclaimable_mb_if_deleted"],
    }, indent=2))


if __name__ == "__main__":
    main()
