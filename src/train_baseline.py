import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features.jsonl"

if not feat_file.exists():
    raise SystemExit("Run feature_extract.py first")

rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]

# Build simple supervised pairs: current frame -> next shuttle xy + pseudo winner
X, y_xy, y_win = [], [], []
for i in range(len(rows) - 1):
    cur = rows[i]
    nxt = rows[i + 1]

    sx = cur["shuttle"]["xy"]
    nx = nxt["shuttle"]["xy"]
    if sx is None or nx is None:
        continue

    pX = cur["players"]["X"]["center"] if cur["players"]["X"] else [0.25, 0.25]
    pY = cur["players"]["Y"]["center"] if cur["players"]["Y"] else [0.75, 0.75]
    sv = cur["shuttle"]["v"]
    sp = cur["shuttle"]["speed"]

    feat = [sx[0], sx[1], sv[0], sv[1], sp, pX[0], pX[1], pY[0], pY[1]]
    X.append(feat)
    y_xy.append(nx)

    # pseudo label: if shuttle in top half next frame => Y attacks, else X attacks
    y_win.append(1 if nx[1] > 0.5 else 0)  # X=1, Y=0 convention can be adapted

X = np.array(X, dtype=float)
y_xy = np.array(y_xy, dtype=float)
y_win = np.array(y_win, dtype=int)

Xtr, Xte, ytr_xy, yte_xy, ytr_w, yte_w = train_test_split(
    X, y_xy, y_win, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(n_estimators=120, random_state=42)
reg.fit(Xtr, ytr_xy)
pred_xy = reg.predict(Xte)
rmse = mean_squared_error(yte_xy, pred_xy) ** 0.5

clf = RandomForestClassifier(n_estimators=120, random_state=42)
clf.fit(Xtr, ytr_w)
pred_w = clf.predict(Xte)
acc = accuracy_score(yte_w, pred_w)

report = {
    "samples": int(len(X)),
    "landing_rmse": float(rmse),
    "winner_acc": float(acc),
    "note": "Baseline with pseudo labels; replace with rally-true labels for real quality."
}

out = ROOT / "reports" / "baseline_metrics.json"
out.write_text(json.dumps(report, indent=2))
print("saved", out)
print(report)
