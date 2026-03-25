from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import json

ROOT = Path(__file__).resolve().parents[1]
new_df = pd.read_csv(ROOT / 'data' / 'quant_features_v5.csv')

feature_cols = [c for c in new_df.columns if c not in ['frame','t_sec','winner_proxy','shuttle_x','shuttle_y']]
X = new_df[feature_cols]
y_cls = new_df['winner_proxy']
y_reg = new_df[['shuttle_x','shuttle_y']]

Xtr, Xte, ytrc, ytec, ytrr, yter = train_test_split(X, y_cls, y_reg, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=320, random_state=42, n_jobs=-1)
clf.fit(Xtr, ytrc)
predc = clf.predict(Xte)
acc = accuracy_score(ytec, predc)

reg = RandomForestRegressor(n_estimators=320, random_state=42, n_jobs=-1)
reg.fit(Xtr, ytrr)
predr = reg.predict(Xte)
rmse = mean_squared_error(yter, predr) ** 0.5

baseline_path = ROOT / 'reports' / 'quant_model_metrics_v3.json'
baseline = {}
if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text())

report = {
  'dataset': 'quant_features_v5.csv (from frame_features_v6)',
  'samples': int(len(new_df)),
  'winner_acc': float(acc),
  'landing_rmse': float(rmse),
  'features': len(feature_cols),
  'compare_vs_v3': {
      'v3_winner_acc': baseline.get('winner_acc'),
      'v3_landing_rmse': baseline.get('landing_rmse'),
      'delta_winner_acc': (float(acc) - float(baseline['winner_acc'])) if baseline.get('winner_acc') is not None else None,
      'delta_landing_rmse': (float(rmse) - float(baseline['landing_rmse'])) if baseline.get('landing_rmse') is not None else None,
  },
  'note': 'v4 compare run on v6 features with same RF config for apples-to-apples.'
}

out = ROOT / 'reports' / 'quant_model_metrics_v4_compare.json'
out.write_text(json.dumps(report, indent=2))
print('saved', out)
print(report)
