from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import json

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'quant_features_v2.csv').sort_values('frame').reset_index(drop=True)

# temporal window length
T = 6
base_cols = [c for c in df.columns if c not in ['frame','t_sec','winner_proxy','shuttle_x','shuttle_y']]

X_rows, y_xy, y_w = [], [], []
for i in range(T, len(df)-1):
    window = df.iloc[i-T:i]
    feat = []
    # concatenate last T frames of selected compact cols
    sel_cols = ['shuttle_speed'] + [c for c in base_cols if c.startswith('X_') or c.startswith('Y_')][:8]
    for _, r in window.iterrows():
        feat.extend([float(r[c]) for c in sel_cols])

    # target = next frame landing + winner_proxy
    nxt = df.iloc[i+1]
    X_rows.append(feat)
    y_xy.append([float(nxt['shuttle_x']), float(nxt['shuttle_y'])])
    y_w.append(int(nxt['winner_proxy']))

X = np.array(X_rows, dtype=float)
y_xy = np.array(y_xy, dtype=float)
y_w = np.array(y_w, dtype=int)

Xtr, Xte, ytr_xy, yte_xy, ytr_w, yte_w = train_test_split(
    X, y_xy, y_w, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(n_estimators=260, random_state=42, n_jobs=-1)
reg.fit(Xtr, ytr_xy)
p_xy = reg.predict(Xte)
rmse = mean_squared_error(yte_xy, p_xy) ** 0.5

clf = RandomForestClassifier(n_estimators=260, random_state=42, n_jobs=-1)
clf.fit(Xtr, ytr_w)
p_w = clf.predict(Xte)
acc = accuracy_score(yte_w, p_w)

report = {
    'samples': int(len(X)),
    'window_T': T,
    'landing_rmse': float(rmse),
    'winner_acc': float(acc),
    'note': 'Temporal window model over motion features.'
}

out = ROOT / 'reports' / 'temporal_model_metrics.json'
out.write_text(json.dumps(report, indent=2))
print('saved', out)
print(report)
