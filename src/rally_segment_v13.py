#!/usr/bin/env python3
"""
Rally Segmentation V13 - Improved sensitivity and coverage
- Lower threshold: single signal enough (OR instead of AND)
- Add shuttle position delta as fallback signal  
- Fill small gaps via interpolation
"""
import csv
import json
import gzip
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

def main():
    print("Loading frame features...")
    # Handle both .jsonl and .jsonl.gz files
    feat_file = DATA / "frame_features_v12.jsonl.gz"
    if feat_file.exists():
        with gzip.open(feat_file, 'rt') as f:
            frames = [json.loads(line) for line in f if line.strip()]
    else:
        feat_file = DATA / "frame_features_v6.jsonl.gz"
        with gzip.open(feat_file, 'rt') as f:
            frames = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total frames: {len(frames)}")
    
    # Extract shuttle positions and player positions
    shuttle_data = []
    player_x_data = []
    player_y_data = []
    
    for f in frames:
        frame = f['frame']
        t_sec = f.get('t_sec', 0)
        
        # Shuttle
        if f.get('shuttle', {}).get('visible'):
            pos = f['shuttle'].get('xy')
            if pos and len(pos) >= 2:
                shuttle_data.append({
                    'frame': frame, 't_sec': t_sec,
                    'x': pos[0], 'y': pos[1],
                    'speed': f['shuttle']['speed']
                })
        
        # Players: 'X' and 'Y' from 'players' key
        players = f.get('players')
        if not players:
            players = {}
        
        px_data = players.get('X') if players else None
        py_data = players.get('Y') if players else None
        
        if px_data and px_data.get('center'):
            px = px_data['center']
            if px and len(px) >= 2:
                player_x_data.append({
                    'frame': frame, 't_sec': t_sec,
                    'x': px[0], 'y': px[1]
                })
        
        if py_data and py_data.get('center'):
            py = py_data['center']
            if py and len(py) >= 2:
                player_y_data.append({
                    'frame': frame, 't_sec': t_sec,
                    'x': py[0], 'y': py[1]
                })
    
    print(f"Valid shuttle positions: {len(shuttle_data)}")
    print(f"Valid player X positions: {len(player_x_data)}")
    print(f"Valid player Y positions: {len(player_y_data)}")
    
    # Build position lookup
    shuttle_lookup = {p['frame']: p for p in shuttle_data}
    player_x_lookup = {p['frame']: p for p in player_x_data}
    player_y_lookup = {p['frame']: p for p in player_y_data}
    
    # Calculate player movement speed (velocity magnitude)
    def calc_movement_speed(player_data):
        if len(player_data) < 2:
            return []
        speeds = []
        for i in range(1, len(player_data)):
            dx = player_data[i]['x'] - player_data[i-1]['x']
            dy = player_data[i]['y'] - player_data[i-1]['y']
            dt = max(player_data[i]['t_sec'] - player_data[i-1]['t_sec'], 0.001)
            speed = np.sqrt(dx*dx + dy*dy) / dt
            speeds.append((player_data[i]['frame'], speed))
        return speeds
    
    player_x_speeds = calc_movement_speed(player_x_data)
    player_y_speeds = calc_movement_speed(player_y_data)
    
    px_speed_lookup = {f: s for f, s in player_x_speeds}
    py_speed_lookup = {f: s for f, s in player_y_speeds}
    
    # Calculate shuttle position delta (additional signal)
    def calc_shuttle_position_delta(data):
        if len(data) < 2:
            return []
        deltas = []
        for i in range(1, len(data)):
            dx = data[i]['x'] - data[i-1]['x']
            dy = data[i]['y'] - data[i-1]['y']
            delta = np.sqrt(dx*dx + dy*dy)
            deltas.append((data[i]['frame'], delta))
        return deltas
    
    shuttle_deltas = calc_shuttle_position_delta(shuttle_data)
    shuttle_delta_lookup = {f: d for f, d in shuttle_deltas}
    
    # Get thresholds - LOWER than v12 for more sensitivity
    shuttle_speeds = np.array([p['speed'] for p in shuttle_data])
    # Use 15th percentile (more inclusive than 25th)
    low_thresh = np.percentile(shuttle_speeds, 15)
    high_thresh = np.percentile(shuttle_speeds, 75)
    
    # Get player movement thresholds - use 30th percentile
    all_px_speeds = [s for f, s in player_x_speeds]
    all_py_speeds = [s for f, s in player_y_speeds]
    if all_px_speeds and all_py_speeds:
        player_low_thresh = np.percentile(all_px_speeds + all_py_speeds, 30)
    else:
        player_low_thresh = 0.1
    
    # Get shuttle delta threshold
    all_deltas = [d for f, d in shuttle_deltas]
    if all_deltas:
        delta_thresh = np.percentile(all_deltas, 20)  # 20th percentile
    else:
        delta_thresh = 5.0
    
    print(f"Shuttle speed: low={low_thresh:.4f}, high={high_thresh:.4f}")
    print(f"Player movement threshold: {player_low_thresh:.4f}")
    print(f"Shuttle delta threshold: {delta_thresh:.4f}")
    
    # V13: Score each frame - single signal enough (OR logic)
    all_frames = sorted(set(p['frame'] for p in shuttle_data))
    
    frame_rally_score = {}
    for frame in all_frames:
        shuttle = shuttle_lookup.get(frame)
        px_speed = px_speed_lookup.get(frame, 0)
        py_speed = py_speed_lookup.get(frame, 0)
        shuttle_delta = shuttle_delta_lookup.get(frame, 0)
        
        # V13: ANY of these signals triggers "in rally"
        score = 0
        if shuttle and shuttle['speed'] > low_thresh:
            score += 1
        if max(px_speed, py_speed) > player_low_thresh:
            score += 1
        if shuttle_delta > delta_thresh:
            score += 1
        
        # V13: Single signal enough (score >= 1)
        frame_rally_score[frame] = 1 if score >= 1 else 0
    
    # Find rallies
    rallies = []
    in_rally = False
    rally_start = None
    
    for frame in all_frames:
        if frame_rally_score.get(frame, 0) >= 1:
            if not in_rally:
                in_rally = True
                rally_start = frame
        else:
            if in_rally:
                in_rally = False
                if frame - rally_start >= 10:  # Min 10 frames (lowered from 15)
                    rallies.append((rally_start, frame))
    
    # Handle last rally
    if in_rally and all_frames[-1] - rally_start >= 10:
        rallies.append((rally_start, all_frames[-1]))
    
    print(f"Found {len(rallies)} rallies (V13 - improved sensitivity)")
    
    # V13: Fill small gaps via interpolation
    # If gap < 10 frames, fill it
    MAX_GAP = 10
    filled_rallies = []
    for i, (s, e) in enumerate(rallies):
        if i == 0:
            filled_rallies.append((s, e))
            continue
        prev_s, prev_e = filled_rallies[-1]
        gap = s - prev_e
        if gap <= MAX_GAP:
            # Fill the gap - merge rallies
            filled_rallies[-1] = (prev_s, e)
        else:
            filled_rallies.append((s, e))
    
    print(f"After gap filling: {len(filled_rallies)} rallies")
    
    # Map frames to rally IDs
    frame_to_rally = {}
    for rid, (start, end) in enumerate(filled_rallies):
        for frame in range(start, end + 1):
            frame_to_rally[frame] = rid
    
    # Print first 10 rallies
    print("First 10 rallies:")
    for i, (s, e) in enumerate(filled_rallies[:10]):
        duration = (e - s) * 0.04
        print(f"  Rally {i}: frames {s}-{e} ({duration:.1f}s)")
    
    # Save to CSV
    out_file = DATA / "rally_labels_v13.csv.gz"
    with gzip.open(out_file, 'wt') as f:
        f.write("frame,rally_id,label\n")
        for frame in all_frames:
            rally_id = frame_to_rally.get(frame, -1)
            label = 1 if rally_id >= 0 else 0
            f.write(f"{frame},{rally_id},{label}\n")
    
    print(f"Saved rally labels to {out_file}")
    
    # Summary
    avg_length = np.mean([e - s + 1 for s, e in filled_rallies]) if filled_rallies else 0
    total_rally_frames = len(frame_to_rally)
    in_rally_pct = total_rally_frames/len(frames)*100
    
    print(f"\nRally segmentation summary (V13 - improved):")
    print(f"  Total rallies: {len(filled_rallies)}")
    print(f"  Average rally length: {avg_length:.1f} frames ({avg_length*0.04:.1f}s)")
    print(f"  Total rally frames: {total_rally_frames}")
    print(f"  Coverage: {in_rally_pct:.1f}%")
    
    # Compare with V12
    try:
        with gzip.open(DATA / "rally_labels_v12.csv.gz", 'rt') as f:
            reader = csv.DictReader(f)
            v12_rows = list(reader)
        v12_rally = sum(1 for r in v12_rows if r['label'] == '1')
        print(f"  vs V12: {v12_rally} rally frames ({v12_rally/len(frames)*100:.1f}%)")
    except Exception as e:
        print(f"  Could not compare with V12: {e}")

if __name__ == "__main__":
    main()
