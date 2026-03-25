import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features_v6.jsonl"
quant_file = ROOT / "data" / "quant_features_v5.csv"
out_file = ROOT / "data" / "rally_labels_v3.csv"
report_file = ROOT / "reports" / f"rally_segmentation_v3_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"

rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]
row_by_frame = {r["frame"]: r for r in rows}

motion_by_frame = {}
if quant_file.exists():
    with quant_file.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            frame = int(float(r["frame"]))
            motion_by_frame[frame] = (
                float(r.get("X_arms_ang_vel", 0) or 0)
                + float(r.get("X_legs_ang_vel", 0) or 0)
                + float(r.get("Y_arms_ang_vel", 0) or 0)
                + float(r.get("Y_legs_ang_vel", 0) or 0)
            )

activity = sorted(motion_by_frame.values())
if activity:
    idx = int(0.70 * (len(activity) - 1))  # lower threshold for higher sensitivity
    motion_th = activity[idx]
else:
    motion_th = 160.0

# More sensitive config to raise rally count
N_GAP = 4
START_STREAK = 1
MIN_RALLY_SPAN = 3

rallies = []
in_rally = False
start = None
last_active_frame = None
inactive = 0
rid = 0

for r in rows:
    f = r["frame"]
    sh = r.get("shuttle", {}) or {}
    vis = bool(sh.get("visible", False))
    move_score = motion_by_frame.get(f, 0.0)
    strong_motion = move_score >= motion_th
    active = vis or strong_motion

    if not in_rally and active:
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
                winner = 1 if y > 0.5 else 0
                rallies.append((rid, start, end, winner, x, y))

            in_rally = False
            start = None
            inactive = 0

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

with out_file.open("w") as f:
    f.write("rally_id,start_frame,end_frame,winner,next_landing_x,next_landing_y\n")
    for row in rallies:
        f.write(",".join(map(str, row)) + "\n")

spans = [r[2] - r[1] for r in rallies]
report = {
    "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    "step_executed": "Tune rally segmentation sensitivity (priority #3)",
    "inputs": {"features": str(feat_file), "motion": str(quant_file)},
    "output": str(out_file),
    "rallies": len(rallies),
    "motion_threshold": round(motion_th, 4),
    "avg_rally_span_frames": round(sum(spans) / len(spans), 2) if spans else 0.0,
    "config": {"N_GAP": N_GAP, "START_STREAK": START_STREAK, "MIN_RALLY_SPAN": MIN_RALLY_SPAN, "motion_quantile": 0.70},
    "next_step": "If rally count >= 8, rerun rally-aware training compare.",
}
report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("saved", out_file)
print("saved", report_file)
print("rallies", len(rallies))
