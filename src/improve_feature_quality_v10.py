#!/usr/bin/env python3
"""
Improve feature extraction v10 - Shot type proxies + stance width
Adds:
- Shuttle direction change rate (for smash/drop shot detection)
- Player stance width (distance between feet)
- Center of gravity estimate
"""
import gzip
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    
    # Input: v9 features with smoothed shuttle
    in_file = DATA / "frame_features_v9.jsonl"
    out_file = DATA / "frame_features_v10.jsonl"
    out_report = REPORTS / f"feature_quality_v10_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
    
    with open(in_file, "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    # Track shuttle trajectory for direction changes
    prev_sh = None
    prev_t = None
    
    for i, r in enumerate(rows):
        t = float(r.get("t_sec", 0))
        sh = r.get("shuttle", {})
        
        # Get shuttle position
        sx = sh.get("x", 0.5)
        sy = sh.get("y", 0.5)
        if sh.get("xy"):
            sx, sy = sh["xy"][0], sh["xy"][1]
        
        # Calculate shuttle direction change (shot type proxy)
        dir_change = 0.0
        if prev_sh is not None and prev_t is not None and t > prev_t:
            dt = t - prev_t
            dx = (sx - prev_sh[0]) / dt
            dy = (sy - prev_sh[1]) / dt
            if i > 1:
                # Compare with previous direction
                prev_dx = (prev_sh[0] - prev_sh[2]) / max(0.001, prev_t - prev_sh[3]) if len(prev_sh) > 2 else 0
                prev_dy = (prev_sh[1] - prev_sh[2]) / max(0.001, prev_t - prev_sh[3]) if len(prev_sh) > 2 else 0
                dir_change = ((dx - prev_dx)**2 + (dy - prev_dy)**2) ** 0.5
        
        r["shuttle_dir_change"] = round(dir_change, 4)
        
        # Add stance width for each player (distance between hip keypoints)
        for side in ["X", "Y"]:
            p = r.get("players", {}).get(side)
            if p and p.get("kpts") and len(p["kpts"]) >= 16:
                kpts = p["kpts"]
                # Left hip (11), Right hip (12) - calculate stance width
                if len(kpts) > 12:
                    l_hip = kpts[11][:2]
                    r_hip = kpts[12][:2]
                    stance_width = ((l_hip[0] - r_hip[0])**2 + (l_hip[1] - r_hip[1])**2) ** 0.5
                    r[f"{side}_stance_width"] = round(stance_width, 4)
                    
                    # Center of gravity estimate (midpoint between hips + shoulder midpoint)
                    if len(kpts) > 5:
                        l_shoulder = kpts[5][:2]
                        r_shoulder = kpts[6][:2]
                        cog_x = (l_hip[0] + r_hip[0] + l_shoulder[0] + r_shoulder[0]) / 4
                        cog_y = (l_hip[1] + r_hip[1] + l_shoulder[1] + r_shoulder[1]) / 4
                        r[f"{side}_cog"] = [round(cog_x, 4), round(cog_y, 4)]
            else:
                r[f"{side}_stance_width"] = 0.0
                r[f"{side}_cog"] = [0.5, 0.5]
        
        prev_sh = [sx, sy, prev_sh[0] if prev_sh else sx, prev_t if prev_t else t] if prev_sh else [sx, sy, sx, t]
        prev_t = t
    
    # Write output
    with open(out_file, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    
    # Calculate quality metrics
    total_frames = len(rows)
    X_stance = [r.get("X_stance_width", 0) for r in rows if r.get("X_stance_width", 0) > 0]
    Y_stance = [r.get("Y_stance_width", 0) for r in rows if r.get("Y_stance_width", 0) > 0]
    
    # Detect rate check - handle None players
    def has_player_kpts(row, side):
        p = row.get("players")
        if not p:
            return False
        side_data = p.get(side)
        if not side_data:
            return False
        return bool(side_data.get("kpts"))
    
    X_detected = sum(1 for r in rows if has_player_kpts(r, "X"))
    Y_detected = sum(1 for r in rows if has_player_kpts(r, "Y"))
    
    # Calculate feature quality score (enhanced from v9)
    detect_score = (X_detected / total_frames + Y_detected / total_frames) / 2
    stance_score = min(1.0, (len(X_stance) + len(Y_stance)) / (2 * total_frames))
    feature_score = detect_score * 0.7 + stance_score * 0.3
    
    quality = round(feature_score * 100, 2)
    
    report = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "step_executed": "Feature extraction v10: Added shuttle direction change (shot type proxy) + player stance width + center of gravity",
        "input": str(in_file.relative_to(ROOT)),
        "output": str(out_file.relative_to(ROOT)),
        "frames": total_frames,
        "feature_quality_score": quality,
        "player_X_detect_rate": round(X_detected / total_frames, 4),
        "player_Y_detect_rate": round(Y_detected / total_frames, 4),
        "new_features": {
            "shuttle_dir_change": "Shot type proxy (high=smash, low=drop/net shot)",
            "stance_width": "Distance between hips - indicates ready position",
            "cog": "Center of gravity estimate from hips + shoulders"
        },
        "v10_changes": {
            "added_shuttle_dir_change": True,
            "added_stance_width": True,
            "added_cog": True,
            "stance_samples": len(X_stance) + len(Y_stance)
        },
        "next_step": "Run quantify_motion_v7 to extract v10 features with angular velocity + new stance/cog features"
    }
    
    out_report.write_text(json.dumps(report, indent=2))
    print(f"Output: {out_file}")
    print(f"Report: {out_report}")
    print(f"Feature quality: {quality}%")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()