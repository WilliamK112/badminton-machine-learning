import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features.jsonl"
quant_file = ROOT / "data" / "quant_features.csv"
out_file = ROOT / "data" / "rally_labels.csv"
report_file = ROOT / "reports" / "rally_segmentation_upgrade_2026-03-24T03-36.json"

rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]

# Load motion summaries (if available) to strengthen rally state when shuttle visibility is sparse.
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

# Build robust motion threshold from data (90th percentile of combined activity)
activity = []
for m in motion_by_frame.values():
    activity.append(m["X_arms"] + m["X_legs"] + m["Y_arms"] + m["Y_legs"])
activity.sort()
if activity:
    idx = int(0.9 * (len(activity) - 1))
    motion_th = activity[idx]
else:
    motion_th = 250.0

# Improved heuristic rally segmentation:
# - Start when shuttle visible OR strong motion sustained for 2 sampled frames.
# - End when neither shuttle visible nor strong motion for N_GAP consecutive sampled frames.
N_GAP = 6
START_STREAK = 2
MIN_RALLY_SPAN = 3

rallies = []
in_rally = False
start = None
last_active_frame = None
inactive = 0
active_streak = 0
rid = 0

for r in rows:
    f = r["frame"]
    sh = r.get("shuttle", {})
    vis = bool(sh.get("visible", False))

    m = motion_by_frame.get(f, {})
    move_score = m.get("X_arms", 0.0) + m.get("X_legs", 0.0) + m.get("Y_arms", 0.0) + m.get("Y_legs", 0.0)
    strong_motion = move_score >= motion_th

    active = vis or strong_motion
    if active:
        active_streak += 1
    else:
        active_streak = 0

    if not in_rally and (vis or active_streak >= START_STREAK):
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

                # pseudo winner by last known shuttle side near rally end
                end_row = next((x for x in rows if x["frame"] == end), None)
                xy = end_row.get("shuttle", {}).get("xy") if end_row else None
                if xy is None:
                    winner = 1
                    lx, ly = 0.5, 0.5
                else:
                    lx, ly = xy
                    winner = 1 if ly > 0.5 else 0  # X=1, Y=0

                rallies.append((rid, start, end, winner, lx, ly))

            in_rally = False
            start = None
            inactive = 0
            active_streak = 0

# flush if needed
if in_rally and start is not None and last_active_frame and (last_active_frame - start) >= MIN_RALLY_SPAN:
    rid += 1
    end = last_active_frame
    end_row = next((x for x in rows if x["frame"] == end), None)
    xy = end_row.get("shuttle", {}).get("xy") if end_row else None
    if xy is None:
        winner = 1
        lx, ly = 0.5, 0.5
    else:
        lx, ly = xy
        winner = 1 if ly > 0.5 else 0
    rallies.append((rid, start, end, winner, lx, ly))

out_file.parent.mkdir(parents=True, exist_ok=True)
with out_file.open("w") as f:
    f.write("rally_id,start_frame,end_frame,winner,next_landing_x,next_landing_y\n")
    for row in rallies:
        f.write(",".join(map(str, row)) + "\n")

spans = [r[2] - r[1] for r in rallies]
report = {
    "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    "step_executed": "Improve rally segmentation (priority #3)",
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
    "next_step": "Retrain model + compare metrics (priority #4).",
}
report_file.parent.mkdir(parents=True, exist_ok=True)
report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("saved", out_file)
print("saved", report_file)
print("rallies", len(rallies))
