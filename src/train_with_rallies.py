import csv
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features.jsonl"
rally_file = ROOT / "data" / "rally_labels.csv"

rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]
by_frame = {r['frame']: r for r in rows}

rallies = []
with rally_file.open() as f:
    rd = csv.DictReader(f)
    for r in rd:
        rallies.append(r)

X, y_xy, y_w = [], [], []
for rr in rallies:
    sf = int(rr['start_frame'])
    ef = int(rr['end_frame'])
    winner = int(rr['winner'])
    lx = float(rr['next_landing_x'])
    ly = float(rr['next_landing_y'])

    # choose latest frame in [sf,ef] with visible shuttle
    rec = None
    for f in range(ef, sf - 1, -1):
        cand = by_frame.get(f)
        if cand and cand.get('shuttle', {}).get('xy') is not None:
            rec = cand
            break
    if rec is None:
        continue

    sxy = rec['shuttle']['xy']

    pX = rec['players']['X']['center'] if rec['players']['X'] else [0.25, 0.25]
    pY = rec['players']['Y']['center'] if rec['players']['Y'] else [0.75, 0.75]
    sv = rec['shuttle']['v']
    sp = rec['shuttle']['speed']

    feat = [sxy[0], sxy[1], sv[0], sv[1], sp, pX[0], pX[1], pY[0], pY[1], (ef-sf)]
    X.append(feat)
    y_xy.append([lx, ly])
    y_w.append(winner)

X = np.array(X, dtype=float)
y_xy = np.array(y_xy, dtype=float)
y_w = np.array(y_w, dtype=int)

if len(X) < 8:
    report = {
        'samples': int(len(X)),
        'note': 'Too few rally samples for robust split; collect more video or lower segmentation gap.'
    }
    out = ROOT / 'reports' / 'rally_metrics.json'
    out.write_text(json.dumps(report, indent=2))
    print('saved', out)
    print(report)
    raise SystemExit(0)

Xtr, Xte, ytr_xy, yte_xy, ytr_w, yte_w = train_test_split(
    X, y_xy, y_w, test_size=0.25, random_state=42
)

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(Xtr, ytr_xy)
pxy = reg.predict(Xte)
rmse = mean_squared_error(yte_xy, pxy) ** 0.5

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(Xtr, ytr_w)
pw = clf.predict(Xte)
acc = accuracy_score(yte_w, pw)

report = {
    'samples': int(len(X)),
    'landing_rmse': float(rmse),
    'winner_acc': float(acc),
    'note': 'Rally-level labels used (heuristic). Replace with manual labels for higher quality.'
}

out = ROOT / 'reports' / 'rally_metrics.json'
out.write_text(json.dumps(report, indent=2))
print('saved', out)
print(report)
