#!/usr/bin/env python3
"""
Improve feature extraction v11 - Shuttle landing prediction + trajectory features
Adds:
- Predicted landing position based on shuttle trajectory (linear extrapolation)
- Shot momentum (rate of change of shuttle speed)
- Court zone encoding (which zone shuttle is in)
"""
import gzip
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

def predict_landing(timeline, current_idx, look_ahead=10):
    """Predict where shuttle will land based on current trajectory."""
    if current_idx >= len(timeline) - 3:
        return None, None
    
    # Get recent points
    recent = timeline[max(0, current_idx-5):current_idx+1]
    if len(recent) < 3:
        return None, None
    
    # Fit linear trend
    times = [p[0] for p in recent]
    xs = [p[1] for p in recent]
    ys = [p[2] for p in recent]
    
    # Simple linear regression
    n = len(times)
    if n < 2:
        return None, None
    
    t_mean = sum(times) / n
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    
    # Slope calculation
    num_x = sum((times[i] - t_mean) * (xs[i] - x_mean) for i in range(n))
    den = sum((times[i] - t_mean) ** 2 for i in range(n))
    vx = num_x / den if den > 0 else 0
    
    num_y = sum((times[i] - t_mean) * (ys[i] - y_mean) for i in range(n))
    vy = num_y / den if den > 0 else 0
    
    # Extrapolate to when shuttle reaches court boundary (y < 0.05 or y > 0.95)
    current_t = times[-1]
    current_y = ys[-1]
    
    if abs(vy) < 0.001:
        return None, None
    
    # Time to reach boundary
    if vy > 0:  # Moving down (toward Y side)
        t_boundary = current_t + (0.95 - current_y) / vy
    else:  # Moving up (toward X side)
        t_boundary = current_t + (0.05 - current_y) / vy
    
    if t_boundary > current_t + 5:  # Too far in future
        return None, None
    
    predicted_x = x_mean + vx * (t_boundary - t_mean)
    predicted_y = 0.95 if vy > 0 else 0.05
    
    return round(predicted_x, 3), round(predicted_y, 3)


def get_court_zone(y):
    """Encode which court zone shuttle is in."""
    if y < 0.2:
        return "front_X"
    elif y < 0.4:
        return "mid_X"
    elif y < 0.6:
        return "net"
    elif y < 0.8:
        return "mid_Y"
    else:
        return "front_Y"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    
    # Input: v12 features
    in_file = DATA / "frame_features_v12.jsonl"
    out_file = DATA / "frame_features_v13.jsonl"
    out_report = REPORTS / f"feature_quality_v13_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
    
    with open(in_file, "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    # Build shuttle timeline for trajectory analysis
    shuttle_timeline = []
    for i, r in enumerate(rows):
        sh = r.get("shuttle", {})
        sx = sh.get("x", 0.5)
        sy = sh.get("y", 0.5)
        if sh.get("xy"):
            sx, sy = sh["xy"][0], sh["xy"][1]
        t = r.get("t_sec", 0)
        shuttle_timeline.append((t, sx, sy))
    
    # Process each frame
    for i, r in enumerate(rows):
        t = r.get("t_sec", 0)
        sh = r.get("shuttle", {})
        sx = sh.get("x", 0.5)
        sy = sh.get("y", 0.5)
        if sh.get("xy"):
            sx, sy = sh["xy"][0], sh["xy"][1]
        
        # Predict landing position
        landing_x, landing_y = predict_landing(shuttle_timeline, i)
        r["predicted_landing_x"] = landing_x if landing_x is not None else -1
        r["predicted_landing_y"] = landing_y if landing_y is not None else -1
        
        # Court zone
        r["court_zone"] = get_court_zone(sy)
        
        # Shot momentum (acceleration)
        if i > 0:
            prev_sh = shuttle_timeline[i-1]
            dt = t - prev_sh[0]
            if dt > 0:
                dx = sx - prev_sh[1]
                dy = sy - prev_sh[2]
                speed = (dx**2 + dy**2) ** 0.5 / dt
                
                # Compare with previous speed
                if i > 1:
                    prev2_sh = shuttle_timeline[i-2]
                    dt2 = prev_sh[0] - prev2_sh[0]
                    if dt2 > 0:
                        prev_dx = prev_sh[1] - prev2_sh[1]
                        prev_dy = prev_sh[2] - prev2_sh[2]
                        prev_speed = (prev_dx**2 + prev_dy**2) ** 0.5 / dt2
                        r["shuttle_momentum"] = round(speed - prev_speed, 4)
                    else:
                        r["shuttle_momentum"] = 0.0
                else:
                    r["shuttle_momentum"] = 0.0
            else:
                r["shuttle_momentum"] = 0.0
        else:
            r["shuttle_momentum"] = 0.0
    
    # Write output
    with open(out_file, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    
    # Calculate quality metrics
    total_frames = len(rows)
    landing_predicted = sum(1 for r in rows if r.get("predicted_landing_x", -1) > 0)
    momentum_available = sum(1 for r in rows if r.get("shuttle_momentum", 0) != 0)
    
    report = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "step_executed": "Feature extraction v13: Added shuttle landing prediction + shot momentum + court zone",
        "input": str(in_file.relative_to(ROOT)),
        "output": str(out_file.relative_to(ROOT)),
        "frames": total_frames,
        "v13_features": {
            "predicted_landing_x": "Predicted X where shuttle will land based on trajectory",
            "predicted_landing_y": "Predicted Y boundary (0.05 or 0.95)",
            "court_zone": "Zone: front_X, mid_X, net, mid_Y, front_Y",
            "shuttle_momentum": "Acceleration of shuttle (speed change rate)"
        },
        "coverage": {
            "landing_predicted_frames": landing_predicted,
            "momentum_available_frames": momentum_available,
            "landing_coverage_pct": round(landing_predicted / total_frames * 100, 1)
        },
        "next_step": "Run quantify_motion to extract v13 features with angular velocity, then retrain model"
    }
    
    out_report.write_text(json.dumps(report, indent=2))
    print(f"Output: {out_file}")
    print(f"Report: {out_report}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()