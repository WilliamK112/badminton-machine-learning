#!/usr/bin/env python3
"""
Rally Segmentation V12 - Enhanced with player position signals
Uses shuttle speed + player movement patterns to find rally boundaries.
"""
import json
import gzip
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

def main():
    print("Loading frame features...")
    with open(DATA / "frame_features_v6.jsonl", 'r') as f:
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
    
    # Get shuttle speed thresholds
    shuttle_speeds = np.array([p['speed'] for p in shuttle_data])
    low_thresh = np.percentile(shuttle_speeds, 25)
    high_thresh = np.percentile(shuttle_speeds, 75)
    
    # Get player movement thresholds (balanced)
    all_px_speeds = [s for f, s in player_x_speeds]
    all_py_speeds = [s for f, s in player_y_speeds]
    if all_px_speeds and all_py_speeds:
        # Use 40th percentile for balanced detection
        player_low_thresh = np.percentile(all_px_speeds + all_py_speeds, 40)
    else:
        player_low_thresh = 0.1
    
    print(f"Shuttle speed: low={low_thresh:.4f}, high={high_thresh:.4f}")
    print(f"Player movement threshold: {player_low_thresh:.4f}")
    
    # Find rally segments using combined signals:
    # Rally = high shuttle speed OR high player movement
    # Reset = low shuttle speed AND low player movement
    
    all_frames = sorted(set(p['frame'] for p in shuttle_data))
    
    # Score each frame for "in rally" likelihood
    # Require at least 2 signals for "in rally"
    frame_rally_score = {}
    for frame in all_frames:
        shuttle = shuttle_lookup.get(frame)
        px_speed = px_speed_lookup.get(frame, 0)
        py_speed = py_speed_lookup.get(frame, 0)
        
        score = 0
        if shuttle:
            if shuttle['speed'] > low_thresh:
                score += 1
            if shuttle['speed'] > high_thresh:
                score += 1
        
        max_player_speed = max(px_speed, py_speed)
        if max_player_speed > player_low_thresh:
            score += 1
        
        # Require at least 2 signals for "in rally"
        frame_rally_score[frame] = 1 if score >= 2 else 0
    
    # Find rallies: consecutive frames with score >= 1
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
                if frame - rally_start >= 15:  # Min 15 frames
                    rallies.append((rally_start, frame))
    
    # Handle last rally
    if in_rally and all_frames[-1] - rally_start >= 15:
        rallies.append((rally_start, all_frames[-1]))
    
    print(f"Found {len(rallies)} rallies (enhanced segmentation)")
    
    # Map frames to rally IDs
    frame_to_rally = {}
    for rid, (start, end) in enumerate(rallies):
        for frame in range(start, end + 1):
            frame_to_rally[frame] = rid
    
    # Print first 10 rallies
    print("First 10 rallies:")
    for i, (s, e) in enumerate(rallies[:10]):
        duration = (e - s) * 0.04
        print(f"  Rally {i}: frames {s}-{e} ({duration:.1f}s)")
    
    # Save to CSV
    out_file = DATA / "rally_labels_v12.csv.gz"
    with gzip.open(out_file, 'wt') as f:
        f.write("frame,rally_id,label\n")
        for frame in all_frames:
            rally_id = frame_to_rally.get(frame, -1)
            label = 1 if rally_id >= 0 else 0
            f.write(f"{frame},{rally_id},{label}\n")
    
    print(f"Saved rally labels to {out_file}")
    
    # Summary
    avg_length = np.mean([e - s + 1 for s, e in rallies]) if rallies else 0
    total_rally_frames = len(frame_to_rally)
    in_rally_pct = total_rally_frames/len(frames)*100
    
    print(f"\nRally segmentation summary (V12 - enhanced):")
    print(f"  Total rallies: {len(rallies)}")
    print(f"  Average rally length: {avg_length:.1f} frames ({avg_length*0.04:.1f}s)")
    print(f"  Total rally frames: {total_rally_frames}")
    print(f"  Coverage: {in_rally_pct:.1f}%")
    
    # Compare with previous version
    try:
        prev_file = DATA / "rally_labels_v11.csv.gz"
        prev_count = 0
        with gzip.open(prev_file, 'rt') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[2] == '1':
                    prev_count += 1
        print(f"  vs V11: {prev_count} rally frames ({prev_count/len(frames)*100:.1f}%)")
    except Exception as e:
        print(f"  Could not compare with V11: {e}")

if __name__ == "__main__":
    main()