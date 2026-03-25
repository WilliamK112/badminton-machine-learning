import csv
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features_v6.jsonl"
rally_file = ROOT / "data" / "rally_labels_v4.csv"
base_metrics_file = ROOT / "reports" / "quant_model_metrics_v4_compare.json"

rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]
by_frame = {r["frame"]: r for r in rows}

rallies = []
with rally_file.open() as f:
    rd = csv.DictReader(f)
    for r in rd:
        rallies.append(r)

X, y_xy, y_w = [], [], []
for rr in rallies:
    sf = int(rr["start_frame"])
    ef = int(rr["end_frame"])
    winner = int(rr["winner"])
    lx = float(rr["next_landing_x"])
    ly = float(rr["next_landing_y"])

    rec = None
    for f in range(ef, sf - 1, -1):
        cand = by_frame.get(f)
        if not cand:
            continue
        sh = cand.get("shuttle", {}) or {}
        x = sh.get("x")
        y = sh.get("y")
        if x is None or y is None:
            xy = sh.get("xy")
            if xy is not None:
                x, y = xy[0], xy[1]
        if x is not None and y is not None:
            rec = cand
            break
    if rec is None:
        continue

    sh = rec.get("shuttle", {}) or {}
    sx = sh.get("x")
    sy = sh.get("y")
    if sx is None or sy is None:
        xy = sh.get("xy") or [0.5, 0.5]
        sx, sy = xy[0], xy[1]

    pX = (rec.get("players", {}).get("X") or {}).get("center") or [0.25, 0.25]
    pY = (rec.get("players", {}).get("Y") or {}).get("center") or [0.75, 0.75]
    sp = float(sh.get("speed", 0.0) or 0.0)

    feat = [sx, sy, sp, pX[0], pX[1], pY[0], pY[1], (ef - sf)]
    X.append(feat)
    y_xy.append([lx, ly])
    y_w.append(winner)

X = np.array(X, dtype=float)
y_xy = np.array(y_xy, dtype=float)
y_w = np.array(y_w, dtype=int)

Xtr, Xte, ytr_xy, yte_xy, ytr_w, yte_w = train_test_split(
    X, y_xy, y_w, test_size=0.25, random_state=42
)

reg = RandomForestRegressor(n_estimators=260, random_state=42)
reg.fit(Xtr, ytr_xy)
pxy = reg.predict(Xte)
rmse = mean_squared_error(yte_xy, pxy) ** 0.5

clf = RandomForestClassifier(n_estimators=260, random_state=42)
clf.fit(Xtr, ytr_w)
pw = clf.predict(Xte)
acc = accuracy_score(yte_w, pw)

base = {}
if base_metrics_file.exists():
    base = json.loads(base_metrics_file.read_text())

report = {
    "dataset": str(rally_file),
    "samples": int(len(X)),
    "winner_acc": float(acc),
    "landing_rmse": float(rmse),
    "note": "rally-aware training on v4 event-segmented labels",
    "compare_vs_v4": {
        "v4_winner_acc": base.get("winner_acc"),
        "v4_landing_rmse": base.get("landing_rmse"),
        "delta_winner_acc": (float(acc) - float(base["winner_acc"])) if base.get("winner_acc") is not None else None,
        "delta_landing_rmse": (float(rmse) - float(base["landing_rmse"])) if base.get("landing_rmse") is not None else None,
    },
}

out = ROOT / "reports" / "rally_metrics_v3_compare.json"
out.write_text(json.dumps(report, indent=2))
print("saved", out)
print(report)
