import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features_v6.jsonl"
out_file = ROOT / "data" / "rally_labels_v4.csv"
report_file = ROOT / "reports" / f"rally_segmentation_v4_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"

rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]

# Build compact shuttle timeline
timeline = []
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
    timeline.append((int(r["frame"]), float(x), float(y)))

# v4 idea: segment by directional events + half-court crossing
# - detect Y-direction sign flips (up/down) as hit-change proxy
# - detect crossings around midline y=0.5 as rally-phase boundaries
# - split long track into multiple rally candidates
MIN_SEG = 8
MAX_SEG = 220
CROSS_BAND = 0.03  # crossing band around midline

boundaries = set()
if timeline:
    boundaries.add(timeline[0][0])

prev = None
prev_vy = None
prev_side = None
for f, x, y in timeline:
    side = 1 if y > 0.5 else 0

    if prev is not None:
        vy = y - prev[2]

        # Direction flip event (ignore tiny jitter)
        if prev_vy is not None and abs(vy) > 0.002 and abs(prev_vy) > 0.002 and (vy * prev_vy < 0):
            boundaries.add(f)

        # Midline crossing event
        crossed = (side != prev_side)
        near_mid = abs(y - 0.5) <= CROSS_BAND or abs(prev[2] - 0.5) <= CROSS_BAND
        if crossed and near_mid:
            boundaries.add(f)

        prev_vy = vy
    else:
        prev_vy = None

    prev = (f, x, y)
    prev_side = side

if timeline:
    boundaries.add(timeline[-1][0])

cuts = sorted(boundaries)
segments = []
for i in range(len(cuts) - 1):
    s = cuts[i]
    e = cuts[i + 1]
    span = e - s
    if span < MIN_SEG:
        continue

    # Split overlong segments to keep trainable granularity
    if span > MAX_SEG:
        k = s
        while k + MAX_SEG < e:
            segments.append((k, k + MAX_SEG))
            k += MAX_SEG
        if e - k >= MIN_SEG:
            segments.append((k, e))
    else:
        segments.append((s, e))

# Deduplicate / merge tiny gaps
clean = []
for s, e in segments:
    if not clean:
        clean.append([s, e])
        continue
    ps, pe = clean[-1]
    if s - pe <= 2:  # tiny gap merge
        clean[-1][1] = max(pe, e)
    else:
        clean.append([s, e])

# Build labels
def shuttle_xy_at_or_before(frame):
    cand = [t for t in timeline if t[0] <= frame]
    if not cand:
        return 0.5, 0.5
    _, x, y = cand[-1]
    return x, y

rows_out = []
rid = 0
for s, e in clean:
    if e - s < MIN_SEG:
        continue
    rid += 1
    lx, ly = shuttle_xy_at_or_before(e)
    winner = 1 if ly > 0.5 else 0
    rows_out.append((rid, s, e, winner, lx, ly))

with out_file.open("w") as f:
    f.write("rally_id,start_frame,end_frame,winner,next_landing_x,next_landing_y\n")
    for r in rows_out:
        f.write(",".join(map(str, r)) + "\n")

spans = [r[2] - r[1] for r in rows_out]
report = {
    "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    "step_executed": "Rally segmentation v4 with direction-flip + midline-cross events",
    "input": str(feat_file),
    "output": str(out_file),
    "rallies": len(rows_out),
    "config": {
        "MIN_SEG": MIN_SEG,
        "MAX_SEG": MAX_SEG,
        "CROSS_BAND": CROSS_BAND,
    },
    "avg_rally_span_frames": round(sum(spans) / len(spans), 2) if spans else 0.0,
    "min_rally_span_frames": min(spans) if spans else 0,
    "max_rally_span_frames": max(spans) if spans else 0,
    "next_step": "If rallies >= 8, retrain rally-aware model using rally_labels_v4.csv.",
}
report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("saved", out_file)
print("saved", report_file)
print("rallies", len(rows_out))
