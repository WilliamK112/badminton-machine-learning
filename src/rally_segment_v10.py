#!/usr/bin/env python3
"""
Rally Segmentation V10 - Fixed version
Uses speed-based gap detection to find natural rally boundaries.
"""
import json
import gzip
import numpy as np
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

def main():
    print("Loading frame features...")
    with open(DATA / "frame_features_v6.jsonl", 'r') as f:
        frames = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total frames: {len(frames)}")
    
    # Extract shuttle positions
    positions = []
    for f in frames:
        if f.get('shuttle', {}).get('visible'):
            pos = f['shuttle'].get('xy')
            if pos and len(pos) >= 2:
                positions.append({
                    'frame': f['frame'],
                    't_sec': f['t_sec'],
                    'x': pos[0],
                    'y': pos[1],
                    'speed': f['shuttle']['speed']
                })
    
    print(f"Valid shuttle positions: {len(positions)}")
    
    # Calculate speed threshold for gap detection
    speeds = np.array([p['speed'] for p in positions])
    low_threshold = np.percentile(speeds, 20)
    high_threshold = np.percentile(speeds, 80)
    
    print(f"Speed thresholds: low={low_threshold:.4f}, high={high_threshold:.4f}")
    
    # Build a mapping from frame -> rally_id
    frame_to_rally = {}
    
    # Find rally segments (high speed periods)
    rallies = []
    in_rally = False
    rally_start = None
    
    for p in positions:
        if p['speed'] > low_threshold:
            if not in_rally:
                in_rally = True
                rally_start = p['frame']
        else:
            if in_rally:
                in_rally = False
                if p['frame'] - rally_start >= 15:  # Min 15 frames (~0.6 sec)
                    rallies.append((rally_start, p['frame']))
    
    # Handle last rally
    if in_rally and positions[-1]['frame'] - rally_start >= 15:
        rallies.append((rally_start, positions[-1]['frame']))
    
    print(f"Found {len(rallies)} rallies")
    
    # Map frames to rally IDs
    for rid, (start, end) in enumerate(rallies):
        for frame in range(start, end + 1):
            frame_to_rally[frame] = rid
    
    # Print first 10 rallies
    print("First 10 rallies:")
    for i, (s, e) in enumerate(rallies[:10]):
        duration = (e - s) * 0.04  # Assuming 25fps
        print(f"  Rally {i}: frames {s}-{e} ({duration:.1f}s)")
    
    # Get all unique frame numbers
    all_frames = sorted(set(p['frame'] for p in positions))
    
    # Save to CSV - each frame gets ONE rally_id (or -1 if not in rally)
    out_file = DATA / "rally_labels_v11.csv.gz"
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
    
    print(f"\nRally segmentation summary:")
    print(f"  Total rallies: {len(rallies)}")
    print(f"  Average rally length: {avg_length:.1f} frames")
    print(f"  Total rally frames: {total_rally_frames}")
    print(f"  Coverage: {total_rally_frames/len(frames)*100:.1f}%")

if __name__ == "__main__":
    main()