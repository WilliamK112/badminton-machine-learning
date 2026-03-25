from pathlib import Path
import csv
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
cur_path = ROOT / 'data' / 'quant_features_v4.csv'
prev_path = ROOT / 'data' / 'quant_features_v3.csv'
out_path = ROOT / 'reports' / 'quant_model_metrics_v4.json'


def resolve_csv_path(path: Path) -> Path:
    if path.exists():
        return path
    gz = Path(f"{path}.gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"Missing input file: {path} (or {gz.name})")


def run_eval(path: Path, n_estimators: int = 360):
    path = resolve_csv_path(path)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    feature_cols = [c for c in rows[0].keys() if c not in ['frame', 't_sec', 'winner_proxy', 'shuttle_x', 'shuttle_y']]

    X = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=float)
    y_cls = np.array([int(float(r['winner_proxy'])) for r in rows], dtype=int)
    y_reg = np.array([[float(r['shuttle_x']), float(r['shuttle_y'])] for r in rows], dtype=float)

    Xtr, Xte, ytrc, ytec, ytrr, yter = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytrc)
    predc = clf.predict(Xte)
    acc = accuracy_score(ytec, predc)

    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    reg.fit(Xtr, ytrr)
    predr = reg.predict(Xte)
    rmse = mean_squared_error(yter, predr) ** 0.5

    return {
        'samples': int(len(rows)),
        'features': int(len(feature_cols)),
        'winner_acc': float(acc),
        'landing_rmse': float(rmse),
    }

cur = run_eval(cur_path)
prev = run_eval(prev_path)

report = {
    'timestamp_local': datetime.now().astimezone().isoformat(timespec='seconds'),
    'step_executed': 'Retrain model + compare metrics (priority #4)',
    'current_dataset': str(cur_path),
    'baseline_dataset': str(prev_path),
    'current': cur,
    'baseline_v3_recomputed': prev,
    'delta': {
        'winner_acc': float(cur['winner_acc'] - prev['winner_acc']),
        'landing_rmse': float(cur['landing_rmse'] - prev['landing_rmse'])
    },
    'next_step': 'Generate visual reports (win-prob timeline + landing heatmap) (priority #5).'
}

out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('saved', out_path)
print(json.dumps(report, indent=2))
