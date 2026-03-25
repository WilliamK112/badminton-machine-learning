#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def open_text_auto(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def resolve_input(base_name: str) -> Path:
    plain = DATA / base_name
    gz = DATA / f"{base_name}.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    raise FileNotFoundError(f"missing input: {plain} or {gz}")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    feat_file = resolve_input("frame_features_v6.jsonl")
    out_file = DATA / "rally_labels_v5.csv.gz"
    report_file = REPORTS / f"rally_segmentation_v5_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"

    with open_text_auto(feat_file, "rt") as f:
        rows = [json.loads(x) for x in f if x.strip()]

    timeline = []
    for r in rows:
        sh = r.get("shuttle", {}) or {}
        x = sh.get("x")
        y = sh.get("y")
        if x is None or y is None:
            xy = sh.get("xy")
            if xy is not None:
                x, y = xy[0], xy[1]
        if x is None or y is None:
            continue
        timeline.append((int(r["frame"]), float(x), float(y)))

    MIN_SEG = 8
    MAX_SEG = 220
    CROSS_BAND = 0.03

    boundaries = set()
    if timeline:
        boundaries.add(timeline[0][0])

    prev = None
    prev_vy = None
    prev_side = None
    for f, x, y in timeline:
        side = 1 if y > 0.5 else 0

        if prev is not None:
            vy = y - prev[2]

            if prev_vy is not None and abs(vy) > 0.002 and abs(prev_vy) > 0.002 and (vy * prev_vy < 0):
                boundaries.add(f)

            crossed = side != prev_side
            near_mid = abs(y - 0.5) <= CROSS_BAND or abs(prev[2] - 0.5) <= CROSS_BAND
            if crossed and near_mid:
                boundaries.add(f)

            prev_vy = vy
        else:
            prev_vy = None

        prev = (f, x, y)
        prev_side = side

    if timeline:
        boundaries.add(timeline[-1][0])

    cuts = sorted(boundaries)
    segments = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        span = e - s
        if span < MIN_SEG:
            continue

        if span > MAX_SEG:
            k = s
            while k + MAX_SEG < e:
                segments.append((k, k + MAX_SEG))
                k += MAX_SEG
            if e - k >= MIN_SEG:
                segments.append((k, e))
        else:
            segments.append((s, e))

    clean = []
    for s, e in segments:
        if not clean:
            clean.append([s, e])
            continue
        ps, pe = clean[-1]
        if s - pe <= 2:
            clean[-1][1] = max(pe, e)
        else:
            clean.append([s, e])

    t_frames = [t[0] for t in timeline]
    t_x = [t[1] for t in timeline]
    t_y = [t[2] for t in timeline]

    def shuttle_xy_at_or_before(frame: int):
        if not t_frames:
            return 0.5, 0.5
        idx = 0
        lo, hi = 0, len(t_frames) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if t_frames[mid] <= frame:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return t_x[idx], t_y[idx]

    rows_out = []
    rid = 0
    for s, e in clean:
        if e - s < MIN_SEG:
            continue
        rid += 1
        lx, ly = shuttle_xy_at_or_before(e)
        winner = 1 if ly > 0.5 else 0
        rows_out.append((rid, s, e, winner, lx, ly))

    with gzip.open(out_file, "wt", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rally_id", "start_frame", "end_frame", "winner", "next_landing_x", "next_landing_y"])
        for r in rows_out:
            wr.writerow(r)

    spans = [r[2] - r[1] for r in rows_out]
    report = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "step_executed": "Reduce storage/runtime (priority #6): rally segmentation now supports .jsonl/.jsonl.gz input and writes compressed-only .csv.gz output",
        "input": str(feat_file.relative_to(ROOT)),
        "output": str(out_file.relative_to(ROOT)),
        "rallies": len(rows_out),
        "config": {
            "MIN_SEG": MIN_SEG,
            "MAX_SEG": MAX_SEG,
            "CROSS_BAND": CROSS_BAND,
        },
        "avg_rally_span_frames": round(sum(spans) / len(spans), 2) if spans else 0.0,
        "min_rally_span_frames": min(spans) if spans else 0,
        "max_rally_span_frames": max(spans) if spans else 0,
        "next_step": "Update rally-aware training scripts to auto-read .csv/.csv.gz labels and consume v5 compressed outputs end-to-end.",
    }
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(str(out_file))
    print(str(report_file))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
