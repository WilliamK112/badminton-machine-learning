#!/usr/bin/env python3
"""
Rally Segmentation V8 - Motion-based with velocity thresholds
Uses shuttle velocity + player movement to detect rally boundaries.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

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


def get_shuttle(frame_data: dict) -> tuple:
    """Extract shuttle (x, y) from frame data."""
    sh = frame_data.get("shuttle", {}) or {}
    
    # Check visible flag first
    if not sh.get("visible", False):
        return None, None
    
    # Try xy first (normalized coordinates)
    xy = sh.get("xy")
    if xy is not None and len(xy) >= 2:
        return float(xy[0]), float(xy[1])
    
    # Try x, y separately
    x = sh.get("x")
    y = sh.get("y")
    if x is not None and y is not None:
        return float(x), float(y)
    
    return None, None


def get_player_position(frame_data: dict, player_key: str) -> tuple:
    """Extract player (x, y) from frame data."""
    # players is a dict with 'X' and 'Y' keys, each containing bbox/center info
    players = frame_data.get("players", {}) or {}
    p = players.get(player_key, {})
    
    if not p:
        return None, None
    
    # Try center first
    center = p.get("center")
    if center and len(center) >= 2:
        return float(center[0]), float(center[1])
    
    # Try bbox
    bbox = p.get("bbox")
    if bbox and len(bbox) >= 4:
        # Use center of bbox
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return cx, cy
    
    return None, None


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Use v13 features (latest)
    feat_file = resolve_input("frame_features_v13.jsonl")
    out_file = DATA / "rally_labels_v10.csv.gz"
    report_file = REPORTS / f"rally_segmentation_v8_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"

    with open_text_auto(feat_file, "rt") as f:
        rows = [json.loads(x) for x in f if x.strip()]

    print(f"Loaded {len(rows)} frames")

    # Build shuttle timeline with velocities
    shuttle_timeline = []
    for i, r in enumerate(rows):
        x, y = get_shuttle(r)
        if x is None or y is None:
            continue
        frame = int(r["frame"])
        
        # Calculate velocity if we have previous position
        vx, vy, speed = 0, 0, 0
        if shuttle_timeline:
            prev = shuttle_timeline[-1]
            dx = x - prev[1]
            dy = y - prev[2]
            vx, vy = dx, dy
            speed = (dx**2 + dy**2) ** 0.5
        
        shuttle_timeline.append((frame, x, y, vx, vy, speed))

    # Build player position timelines
    player_timelines = {"X": [], "Y": []}
    for r in rows:
        for p in ["X", "Y"]:
            x, y = get_player_position(r, p)
            if x is not None:
                player_timelines[p].append((int(r["frame"]), x, y))

    print(f"Shuttle timeline: {len(shuttle_timeline)} points")
    print(f"Player X: {len(player_timelines['X'])} points")
    print(f"Player Y: {len(player_timelines['Y'])} points")

    # Find rallies based on shuttle motion
    # Rally starts when shuttle starts moving fast
    # Rally ends when shuttle stops moving
    
    # Compute speed threshold based on data
    speeds = [s[5] for s in shuttle_timeline]
    speed_p10 = sorted(speeds)[int(len(speeds) * 0.1)] if speeds else 0.01
    speed_p90 = sorted(speeds)[int(len(speeds) * 0.9)] if speeds else 0.1
    
    rallies = []
    in_rally = False
    rally_start = None
    
    # Parameters - use data-driven thresholds
    SHUTTLE_IDLE_SPEED = speed_p10  # Bottom 10% = idle
    SHUTTLE_ACTIVE_SPEED = speed_p90  # Top 10% = active rally
    RALLY_MIN_FRAMES = 10
    
    print(f"Speed thresholds: idle < {SHUTTLE_IDLE_SPEED:.4f}, active > {SHUTTLE_ACTIVE_SPEED:.4f}")
    
    for i, (frame, x, y, vx, vy, speed) in enumerate(shuttle_timeline):
        if not in_rally:
            # Look for rally start - shuttle moving after being idle
            if i > 0 and speed > SHUTTLE_IDLE_SPEED:
                # Check previous frame was idle
                prev_speed = shuttle_timeline[i-1][5] if i > 0 else 0
                if prev_speed < SHUTTLE_IDLE_SPEED:
                    in_rally = True
                    rally_start = frame
        else:
            # Look for rally end - shuttle stopped for a while
            if speed < SHUTTLE_IDLE_SPEED:
                # Count consecutive idle frames
                idle_count = 0
                for j in range(i, min(i+10, len(shuttle_timeline))):
                    if shuttle_timeline[j][5] < SHUTTLE_IDLE_SPEED:
                        idle_count += 1
                    else:
                        break
                
                if idle_count >= 3:  # At least 3 consecutive idle frames
                    rally_end = frame
                    if rally_end - rally_start >= RALLY_MIN_FRAMES:
                        rallies.append((rally_start, rally_end))
                    in_rally = False
                    rally_start = None
    
    # Handle case where rally extends to end
    if in_rally:
        rally_end = shuttle_timeline[-1][0]
        if rally_end - rally_start >= RALLY_MIN_FRAMES:
            rallies.append((rally_start, rally_end))

    print(f"Found {len(rallies)} rallies")

    # Also detect based on player position changes (large jumps)
    player_rallies = []
    for p in ["X", "Y"]:
        pts = player_timelines[p]
        for i in range(1, len(pts)):
            dx = abs(pts[i][1] - pts[i-1][1])
            dy = abs(pts[i][2] - pts[i-1][2])
            dist = (dx**2 + dy**2)**0.5
            
            # Large jump = rally boundary
            if dist > 100:  # 100 pixel jump
                player_rallies.append((pts[i-1][0], pts[i][0]))

    # Merge detections
    print(f"Player boundary candidates: {len(player_rallies)}")

    # Output rallies to CSV
    with gzip.open(out_file, "wt") as f:
        f.write("frame,rally_id,label\n")
        for rid, (start, end) in enumerate(rallies):
            for frame in range(start, end + 1):
                f.write(f"{frame},{rid},1\n")

    # Save report
    report = {
        "method": "velocity_threshold",
        "shuttle_idle_speed": SHUTTLE_IDLE_SPEED,
        "rally_min_frames": RALLY_MIN_FRAMES,
        "total_rallies": len(rallies),
        "rallies": [{"start": s, "end": e} for s, e in rallies],
        "frames_with_shuttle": len(shuttle_timeline),
        "frames_with_player_X": len(player_timelines["X"]),
        "frames_with_player_Y": len(player_timelines["Y"]),
    }
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved {len(rallies)} rallies to {out_file}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
