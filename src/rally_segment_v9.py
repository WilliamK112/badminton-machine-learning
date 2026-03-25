#!/usr/bin/env python3
"""
Rally Segmentation V9 - Simple court zone based
Uses shuttle court position to detect rally boundaries.
"""
from __future__ import annotations

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

    # Use v13 features (latest)
    feat_file = resolve_input("frame_features_v13.jsonl")
    out_file = DATA / "rally_labels_v10.csv.gz"
    report_file = REPORTS / f"rally_segmentation_v9_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"

    with open_text_auto(feat_file, "rt") as f:
        rows = [json.loads(x) for x in f if x.strip()]

    print(f"Loaded {len(rows)} frames")

    # Extract shuttle position + court zone
    data_points = []
    for r in rows:
        sh = r.get("shuttle", {})
        if not sh.get("visible"):
            continue
        
        xy = sh.get("xy")
        if not xy or len(xy) < 2:
            continue
        
        x, y = xy[0], xy[1]
        zone = r.get("court_zone", "unknown")
        frame = int(r["frame"])
        
        data_points.append({
            "frame": frame,
            "x": x,
            "y": y,
            "zone": zone,
            "speed": sh.get("speed", 0)
        })

    print(f"Valid data points: {len(data_points)}")

    # Simple segmentation: Find periods where shuttle is moving (rally in progress)
    # Rally = sequence of frames where shuttle speed > threshold
    
    # Use median speed as threshold
    speeds = [d["speed"] for d in data_points]
    speeds_sorted = sorted(speeds)
    speed_threshold = speeds_sorted[len(speeds_sorted) // 3]  # bottom third = "idle"
    
    print(f"Speed threshold (idle): {speed_threshold:.4f}")

    # Group consecutive moving frames into rallies
    rallies = []
    current_rally = []
    
    for d in data_points:
        if d["speed"] > speed_threshold:
            current_rally.append(d["frame"])
        else:
            if current_rally:
                if len(current_rally) >= 10:  # Min 10 frames
                    rallies.append((current_rally[0], current_rally[-1]))
                current_rally = []
    
    # Don't forget last rally
    if current_rally and len(current_rally) >= 10:
        rallies.append((current_rally[0], current_rally[-1]))

    print(f"Found {len(rallies)} rallies")
    for i, (s, e) in enumerate(rallies):
        print(f"  Rally {i}: frames {s} to {e} ({e-s+1} frames)")

    # Output rallies to CSV
    with gzip.open(out_file, "wt") as f:
        f.write("frame,rally_id,label\n")
        for rid, (start, end) in enumerate(rallies):
            for frame in range(start, end + 1):
                f.write(f"{frame},{rid},1\n")

    # Save report
    report = {
        "method": "speed_threshold",
        "speed_threshold": speed_threshold,
        "total_rallies": len(rallies),
        "rallies": [{"start": s, "end": e, "length": e-s+1} for s, e in rallies],
        "frames_analyzed": len(data_points),
    }
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved {len(rallies)} rallies to {out_file}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
