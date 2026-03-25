#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


TZ = ZoneInfo("America/Chicago")


def now_ts() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%dT%H-%M-%S%z")


def load_latest_plan(reports_dir: Path) -> dict:
    plans = sorted(reports_dir.glob("stale_artifact_cleanup_plan_v1_*.json"))
    if not plans:
        raise FileNotFoundError("No cleanup plan found in reports/.")
    latest = plans[-1]
    return json.loads(latest.read_text()), latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded stale artifact cleanup runner")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="Actually delete candidates (default: dry-run)")
    parser.add_argument(
        "--allow-prefix",
        action="append",
        default=["data/"],
        help="Allowed relative path prefixes for deletion",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    reports_dir = root / "reports"

    plan, plan_path = load_latest_plan(reports_dir)

    dry_run = not args.apply
    allow_prefixes = tuple(args.allow_prefix)

    deleted = []
    skipped = []
    reclaimed = 0

    for c in plan.get("candidates", []):
        rel = c["path"]
        twin_rel = c.get("compressed_twin")
        p = root / rel
        twin = root / twin_rel if twin_rel else None

        if not rel.startswith(allow_prefixes):
            skipped.append({"path": rel, "reason": "outside_allow_prefix"})
            continue
        if not p.exists():
            skipped.append({"path": rel, "reason": "missing_candidate"})
            continue
        if not twin or not twin.exists():
            skipped.append({"path": rel, "reason": "missing_compressed_twin"})
            continue

        size = p.stat().st_size
        if dry_run:
            deleted.append({"path": rel, "size_bytes": size, "action": "would_delete"})
            reclaimed += size
        else:
            p.unlink()
            deleted.append({"path": rel, "size_bytes": size, "action": "deleted"})
            reclaimed += size

    out = {
        "generated_at": datetime.now(TZ).isoformat(),
        "step": "guarded_stale_artifact_cleanup_runner",
        "dry_run": dry_run,
        "plan_source": str(plan_path.relative_to(root)),
        "allow_prefixes": list(allow_prefixes),
        "candidates_in_plan": len(plan.get("candidates", [])),
        "handled_count": len(deleted),
        "skipped_count": len(skipped),
        "reclaimed_bytes": reclaimed,
        "reclaimed_mb": round(reclaimed / 1024 / 1024, 2),
        "handled": deleted,
        "skipped": skipped,
    }

    out_path = reports_dir / f"stale_artifact_cleanup_run_v1_{now_ts()}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
