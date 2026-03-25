from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import json

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'quant_features_v2.csv')

feature_cols = [c for c in df.columns if c not in ['frame','t_sec','winner_proxy','shuttle_x','shuttle_y']]
X = df[feature_cols]
y_cls = df['winner_proxy']
y_reg = df[['shuttle_x','shuttle_y']]

Xtr, Xte, ytrc, ytec, ytrr, yter = train_test_split(X, y_cls, y_reg, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=260, random_state=42)
clf.fit(Xtr, ytrc)
predc = clf.predict(Xte)
acc = accuracy_score(ytec, predc)

reg = RandomForestRegressor(n_estimators=260, random_state=42)
reg.fit(Xtr, ytrr)
predr = reg.predict(Xte)
rmse = mean_squared_error(yter, predr) ** 0.5

report = {
  'samples': int(len(df)),
  'winner_acc': float(acc),
  'landing_rmse': float(rmse),
  'features': len(feature_cols),
  'note': 'v2 features with improved shuttle visibility.'
}

out = ROOT / 'reports' / 'quant_model_metrics_v2.json'
out.write_text(json.dumps(report, indent=2))
print('saved', out)
print(report)
