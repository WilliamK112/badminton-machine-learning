#!/usr/bin/env python3
"""Train v15: Combine angular velocity + endpoint + motion features from rally_v9"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]

# Load rally labels with motion features
with gzip.open(ROOT / 'data/rally_labels_v9.csv.gz', 'rt') as f:
    rally_v9 = pd.read_csv(f)
print(f"Loaded {len(rally_v9)} rallies with motion features")

# Load quant features
with gzip.open(ROOT / 'data/quant_features_v11.csv.gz', 'rt') as f:
    quant = pd.read_csv(f)
print(f"Loaded {len(quant)} frame features")

# Build rally-level features
features_list = []
for _, row in rally_v9.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    rally_frames = quant[(quant['frame'] >= start) & (quant['frame'] <= end)]
    
    if len(rally_frames) < 3:
        continue
    
    # Endpoint features (from train_v13)
    end_row = rally_frames.iloc[-1]
    shuttle_x_end = end_row['shuttle_x']
    shuttle_y_end = end_row['shuttle_y']
    dist_from_center = abs(shuttle_x_end - 0.5)
    is_deep = 1 if shuttle_y_end > 0.7 else 0
    
    # Angular velocity stats (from v11/v12)
    feat_dict = {
        'rally_idx': len(features_list),
        'winner': int(row['winner']),
        'shuttle_x_end': shuttle_x_end,
        'shuttle_y_end': shuttle_y_end,
        'dist_from_center': dist_from_center,
        'is_deep': is_deep,
        'avg_motion': row['avg_motion'],
        'max_motion': row['max_motion'],
        'frame_count': row['frame_count'],
    }
    
    for player in ['X', 'Y']:
        for joint in ['arms', 'torso', 'legs']:
            col = f"{player}_{joint}_ang_vel"
            if col in rally_frames.columns:
                vals = rally_frames[col].dropna()
                feat_dict[f'{player}_{joint}_ang_vel_mean'] = vals.mean() if len(vals) else 0
                feat_dict[f'{player}_{joint}_ang_vel_max'] = vals.max() if len(vals) else 0
    
    features_list.append(feat_dict)

df = pd.DataFrame(features_list)
print(f"Built {len(df)} rally samples with features")

# Feature columns (exclude rally_idx, winner)
feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].values
y = df['winner'].values

# Handle NaN values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Class distribution: winner=1: {sum(y==1)}, winner=0: {sum(y==0)}")

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42+fold)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = balanced_accuracy_score(y_test, pred)
    cv_scores.append(score)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"CV balanced accuracy: {cv_mean:.3f} (+/- {cv_std:.3f})")

# Train final model on all data
model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
model.fit(X, y)

# Temporal test (last 8 samples)
test_size = 8
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

model_test = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
model_test.fit(X_train, y_train)
y_pred = model_test.predict(X_test)

temporal_acc = accuracy_score(y_test, y_pred)
temporal_bal = balanced_accuracy_score(y_test, y_pred)
temporal_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
temporal_mcc = matthews_corrcoef(y_test, y_pred)

print(f"Temporal test ({test_size} samples): acc={temporal_acc:.3f}, balanced={temporal_bal:.3f}")

# Feature importance
importances = dict(zip(feature_cols, model.feature_importances_))
top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:8])
print(f"Top features: {top_features}")

# Save report
report = {
    'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
    'step_executed': 'Train v15 - Combine angular velocity + endpoint + motion features',
    'samples': len(df),
    'train': len(df) - test_size,
    'test': test_size,
    'features': len(feature_cols),
    'feature_names': feature_cols,
    'class_distribution': {'winner_1': int(sum(y==1)), 'winner_0': int(sum(y==0))},
    'model': 'GradientBoosting',
    'cv_balanced_accuracy': round(cv_mean, 3),
    'cv_std': round(cv_std, 3),
    'temporal_test': {
        'accuracy': round(temporal_acc, 3),
        'balanced_accuracy': round(temporal_bal, 3),
        'f1_macro': round(temporal_f1, 3),
        'mcc': round(temporal_mcc, 3)
    },
    'top_features': {k: round(v, 3) for k, v in top_features.items()},
    'next_step': 'Compare with v13/v14 metrics, try stance features'
}

out_path = ROOT / 'reports' / f"train_v15_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"Saved: {out_path}")
