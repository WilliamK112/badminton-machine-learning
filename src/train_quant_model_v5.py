from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def resolve_quant_path() -> Path:
    preferred = ROOT / 'data' / 'quant_features_v6.csv.gz'
    fallback_gz = ROOT / 'data' / 'quant_features_v5.csv.gz'
    fallback_csv = ROOT / 'data' / 'quant_features_v5.csv'
    if preferred.exists():
        return preferred
    if fallback_gz.exists():
        return fallback_gz
    if fallback_csv.exists():
        return fallback_csv
    raise FileNotFoundError('Missing quant features input (checked v6.csv.gz, v5.csv.gz, v5.csv).')


in_path = resolve_quant_path()
new_df = pd.read_csv(in_path, compression='infer')

feature_cols = [c for c in new_df.columns if c not in ['frame', 't_sec', 'winner_proxy', 'shuttle_x', 'shuttle_y']]
X = new_df[feature_cols]
y_cls = new_df['winner_proxy']
y_reg = new_df[['shuttle_x', 'shuttle_y']]

Xtr, Xte, ytrc, ytec, ytrr, yter = train_test_split(X, y_cls, y_reg, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=320, random_state=42, n_jobs=-1)
clf.fit(Xtr, ytrc)
predc = clf.predict(Xte)
acc = accuracy_score(ytec, predc)

reg = RandomForestRegressor(n_estimators=320, random_state=42, n_jobs=-1)
reg.fit(Xtr, ytrr)
predr = reg.predict(Xte)
rmse = mean_squared_error(yter, predr) ** 0.5

baseline_path = ROOT / 'reports' / 'quant_model_metrics_v4_compare.json'
baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

report = {
    'timestamp_local': datetime.now().astimezone().isoformat(timespec='seconds'),
    'step_executed': 'Retrain model + compare metrics (compressed-first follow-up)',
    'dataset': str(in_path.relative_to(ROOT)),
    'samples': int(len(new_df)),
    'winner_acc': float(acc),
    'landing_rmse': float(rmse),
    'features': int(len(feature_cols)),
    'compare_vs_v4_compare': {
        'v4_winner_acc': baseline.get('winner_acc'),
        'v4_landing_rmse': baseline.get('landing_rmse'),
        'delta_winner_acc': (float(acc) - float(baseline['winner_acc'])) if baseline.get('winner_acc') is not None else None,
        'delta_landing_rmse': (float(rmse) - float(baseline['landing_rmse'])) if baseline.get('landing_rmse') is not None else None,
    },
    'next_step': 'Apply compressed-first input fallback to remaining report entrypoints and then safely clean stale uncompressed artifacts.',
    'note': 'Uses gzip input by default to avoid creating/depending on duplicate plain CSV files.'
}

out = ROOT / 'reports' / 'quant_model_metrics_v5.json'
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('saved', out)
print(json.dumps(report, indent=2))
