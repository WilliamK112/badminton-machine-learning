#!/usr/bin/env python3
"""
Rally Segmentation V6 - Improved with player position cues
Uses shuttle motion + player position changes to detect rally boundaries.
"""
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


def get_player_position(frame_data: dict, player_key: str) -> tuple:
    """Extract player (x, y) from frame data."""
    p = frame_data.get(player_key, {}) or {}
    if isinstance(p, list) and len(p) >= 2:
        return float(p[0]), float(p[1])
    if isinstance(p, dict):
        x = p.get("x") or (p.get("xy")[0] if p.get("xy") else None)
        y = p.get("y") or (p.get("xy")[1] if p.get("xy") else None)
        if x is not None and y is not None:
            return float(x), float(y)
    return None, None


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    feat_file = resolve_input("frame_features_v6.jsonl")
    out_file = DATA / "rally_labels_v6.csv.gz"
    report_file = REPORTS / f"rally_segmentation_v6_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"

    with open_text_auto(feat_file, "rt") as f:
        rows = [json.loads(x) for x in f if x.strip()]

    # Build shuttle timeline
    shuttle_timeline = []
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
        shuttle_timeline.append((int(r["frame"]), float(x), float(y)))

    # Build player position timelines
    player_x_timeline = []
    player_y_timeline = []
    for r in rows:
        fx, fy = get_player_position(r, "player_X")
        bx, by = get_player_position(r, "player_Y")
        frame = int(r["frame"])
        if fx is not None:
            player_x_timeline.append((frame, fx, fy))
        if bx is not None:
            player_y_timeline.append((frame, bx, by))

    # Configuration
    MIN_SEG = 8
    MAX_SEG = 200  # Slightly tighter max
    CROSS_BAND = 0.03
    PLAYER_MOVE_THRESHOLD = 0.08  # Player moved significantly

    boundaries = set()
    
    # Always start at first frame
    if shuttle_timeline:
        boundaries.add(shuttle_timeline[0][0])

    # ==== Shuttle-based boundary detection ====
    prev = None
    prev_vy = None
    prev_side = None
    for f, x, y in shuttle_timeline:
        side = 1 if y > 0.5 else 0

        if prev is not None:
            vy = y - prev[2]

            # Velocity reversal = likely rally change
            if prev_vy is not None and abs(vy) > 0.002 and abs(prev_vy) > 0.002 and (vy * prev_vy < 0):
                boundaries.add(f)

            # Cross-court near mid = rally boundary
            crossed = side != prev_side
            near_mid = abs(y - 0.5) <= CROSS_BAND or abs(prev[2] - 0.5) <= CROSS_BAND
            if crossed and near_mid:
                boundaries.add(f)

            prev_vy = vy
        else:
            prev_vy = None

        prev = (f, x, y)
        prev_side = side

    # ==== Player position change detection ====
    # When players move significantly toward net, rally likely starting
    # When players retreat, rally likely ending
    
    def detect_player_movement(timeline: list, window: int = 15) -> list:
        """Detect frames where player makes significant position change."""
        moves = []
        for i in range(len(timeline) - window):
            f1, x1, y1 = timeline[i]
            f2, x2, y2 = timeline[i + window]
            dist = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            if dist > PLAYER_MOVE_THRESHOLD:
                moves.append((f1, f2, dist))
        return moves

    # Add player movement events as potential boundaries
    for timeline in [player_x_timeline, player_y_timeline]:
        if len(timeline) > 15:
            moves = detect_player_movement(timeline)
            for f1, f2, dist in moves:
                # Add midpoint of movement as potential boundary
                boundaries.add((f1 + f2) // 2)

    # End at last frame
    if shuttle_timeline:
        boundaries.add(shuttle_timeline[-1][0])

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

    # Merge adjacent segments
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

    # Build shuttle lookup
    t_frames = [t[0] for t in shuttle_timeline]
    t_x = [t[1] for t in shuttle_timeline]
    t_y = [t[2] for t in shuttle_timeline]

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

    # Generate output with improved winner detection
    rows_out = []
    rid = 0
    for s, e in clean:
        if e - s < MIN_SEG:
            continue
        rid += 1
        
        # Use shuttle position at end of rally + motion to predict winner
        lx, ly = shuttle_xy_at_or_before(e)
        
        # Check shuttle direction at end (moving toward X side or Y side?)
        # Look at last few frames to determine direction
        end_frames = [f for f, _, _ in shuttle_timeline if s <= f <= e]
        if len(end_frames) >= 3:
            last_3 = [(x, y) for f, x, y in shuttle_timeline if s <= f <= e][-3:]
            # If y decreasing (moving up), going toward X (top court)
            # If y increasing (moving down), going toward Y (bottom court)
            y_trend = last_3[-1][1] - last_3[0][1]
            # If shuttle moving up (y decreasing) and ends in top half -> X wins
            # If shuttle moving down (y increasing) and ends in bottom half -> Y wins
            winner = 1 if ly > 0.5 else 0
        else:
            winner = 1 if ly > 0.5 else 0
            
        rows_out.append((rid, s, e, winner, lx, ly))

    with gzip.open(out_file, "wt", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rally_id", "start_frame", "end_frame", "winner", "next_landing_x", "next_landing_y"])
        for r in rows_out:
            wr.writerow(r)

    spans = [r[2] - r[1] for r in rows_out]
    
    # Calculate class distribution
    winners = [r[3] for r in rows_out]
    winner_0 = winners.count(0)
    winner_1 = winners.count(1)
    
    report = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "step_executed": "Improve rally segmentation (priority #3): Added player position change detection",
        "input": str(feat_file.relative_to(ROOT)),
        "output": str(out_file.relative_to(ROOT)),
        "rallies": len(rows_out),
        "class_distribution": {
            "winner_0_X_court": winner_0,
            "winner_1_Y_court": winner_1,
            "imbalance_ratio": round(winner_0 / winner_1, 2) if winner_1 > 0 else "inf"
        },
        "config": {
            "MIN_SEG": MIN_SEG,
            "MAX_SEG": MAX_SEG,
            "CROSS_BAND": CROSS_BAND,
            "PLAYER_MOVE_THRESHOLD": PLAYER_MOVE_THRESHOLD,
            "player_based_detection": True
        },
        "avg_rally_span_frames": round(sum(spans) / len(spans), 2) if spans else 0.0,
        "min_rally_span_frames": min(spans) if spans else 0,
        "max_rally_span_frames": max(spans) if spans else 0,
        "next_step": "Retrain model with v6 rally labels and compare metrics",
    }
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(str(out_file))
    print(str(report_file))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
