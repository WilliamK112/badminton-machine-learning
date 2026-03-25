import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features_v6.jsonl"
quant_file = ROOT / "data" / "quant_features_v5.csv"
out_file = ROOT / "data" / "rally_labels_v2.csv"
report_file = ROOT / "reports" / f"rally_segmentation_v2_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"

rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]
row_by_frame = {r["frame"]: r for r in rows}

# Motion summary by frame
motion_by_frame = {}
if quant_file.exists():
    with quant_file.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            frame = int(float(r["frame"]))
            motion_by_frame[frame] = {
                "X_arms": float(r.get("X_arms_ang_vel", 0) or 0),
                "X_legs": float(r.get("X_legs_ang_vel", 0) or 0),
                "Y_arms": float(r.get("Y_arms_ang_vel", 0) or 0),
                "Y_legs": float(r.get("Y_legs_ang_vel", 0) or 0),
            }

# robust threshold (85th percentile to improve sensitivity)
activity = [m["X_arms"] + m["X_legs"] + m["Y_arms"] + m["Y_legs"] for m in motion_by_frame.values()]
activity.sort()
if activity:
    idx = int(0.85 * (len(activity) - 1))
    motion_th = activity[idx]
else:
    motion_th = 220.0

# v2 segmentation config
N_GAP = 8
START_STREAK = 3
MIN_RALLY_SPAN = 6

rallies = []
in_rally = False
start = None
last_active_frame = None
inactive = 0
active_streak = 0
rid = 0

for r in rows:
    f = r["frame"]
    sh = r.get("shuttle", {}) or {}
    vis = bool(sh.get("visible", False))

    m = motion_by_frame.get(f, {})
    move_score = m.get("X_arms", 0.0) + m.get("X_legs", 0.0) + m.get("Y_arms", 0.0) + m.get("Y_legs", 0.0)
    strong_motion = move_score >= motion_th

    # Start condition is stricter: require sustained activity
    active = vis or strong_motion
    active_streak = active_streak + 1 if active else 0

    if not in_rally and active_streak >= START_STREAK:
        in_rally = True
        start = f
        last_active_frame = f
        inactive = 0
        continue

    if in_rally:
        if active:
            last_active_frame = f
            inactive = 0
        else:
            inactive += 1

        if inactive >= N_GAP:
            end = last_active_frame if last_active_frame is not None else f
            if end - start >= MIN_RALLY_SPAN:
                rid += 1
                end_row = row_by_frame.get(end, {})
                end_sh = end_row.get("shuttle", {}) if end_row else {}
                x = end_sh.get("x")
                y = end_sh.get("y")
                if x is None or y is None:
                    xy = end_sh.get("xy") or [0.5, 0.5]
                    x, y = xy[0], xy[1]
                winner = 1 if y > 0.5 else 0  # X=1, Y=0 proxy
                rallies.append((rid, start, end, winner, x, y))

            in_rally = False
            start = None
            inactive = 0
            active_streak = 0

# flush tail rally
if in_rally and start is not None and last_active_frame and (last_active_frame - start) >= MIN_RALLY_SPAN:
    rid += 1
    end = last_active_frame
    end_row = row_by_frame.get(end, {})
    end_sh = end_row.get("shuttle", {}) if end_row else {}
    x = end_sh.get("x")
    y = end_sh.get("y")
    if x is None or y is None:
        xy = end_sh.get("xy") or [0.5, 0.5]
        x, y = xy[0], xy[1]
    winner = 1 if y > 0.5 else 0
    rallies.append((rid, start, end, winner, x, y))

out_file.parent.mkdir(parents=True, exist_ok=True)
with out_file.open("w") as f:
    f.write("rally_id,start_frame,end_frame,winner,next_landing_x,next_landing_y\n")
    for row in rallies:
        f.write(",".join(map(str, row)) + "\n")

spans = [r[2] - r[1] for r in rallies]
report = {
    "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    "step_executed": "Improve rally segmentation (priority #3) on v6/v5 features",
    "inputs": {
        "features": str(feat_file),
        "motion": str(quant_file) if quant_file.exists() else None,
    },
    "output": str(out_file),
    "rallies": len(rallies),
    "motion_threshold": round(motion_th, 4),
    "avg_rally_span_frames": round(sum(spans) / len(spans), 2) if spans else 0.0,
    "max_rally_span_frames": max(spans) if spans else 0,
    "min_rally_span_frames": min(spans) if spans else 0,
    "config": {
        "N_GAP": N_GAP,
        "START_STREAK": START_STREAK,
        "MIN_RALLY_SPAN": MIN_RALLY_SPAN,
    },
    "next_step": "Retrain model with rally-aware labels and compare metrics.",
}
report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("saved", out_file)
print("saved", report_file)
print("rallies", len(rallies))
