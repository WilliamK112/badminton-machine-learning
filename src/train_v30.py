#!/usr/bin/env python3
"""Train v30: All features + extensive RF hyperparameter search"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]

# Load rally labels
with gzip.open(ROOT / 'data/rally_labels_v9.csv.gz', 'rt') as f:
    rally_v9 = pd.read_csv(f)
print(f"Loaded {len(rally_v9)} rallies")

# Load quant features (v11 has angular velocity)
with gzip.open(ROOT / 'data/quant_features_v11.csv.gz', 'rt') as f:
    quant = pd.read_csv(f)
print(f"Loaded {len(quant)} frame quant features")

# Load frame features v12
import json as json_lib
frame_features = []
with open(ROOT / 'data/frame_features_v12.jsonl', 'r') as f:
    for line in f:
        frame_features.append(json_lib.loads(line))
frame_df = pd.DataFrame(frame_features)
print(f"Loaded {len(frame_df)} frame features")

# Build rally-level features - use ALL features
features_list = []
for _, row in rally_v9.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    rally_frames = quant[(quant['frame'] >= start) & (quant['frame'] <= end)]
    rally_frames_frame = frame_df[(frame_df['frame'] >= start) & (frame_df['frame'] <= end)]
    
    if len(rally_frames) < 3:
        continue
    
    feat_dict = {'rally_idx': len(features_list), 'winner': int(row['winner'])}
    
    # --- Quant features: mean, std, max, min, range for each ---
    quant_cols = ['shuttle_x', 'shuttle_y', 'shuttle_speed', 'shuttle_dir_change',
                  'X_arms_ang_vel', 'X_torso_ag_vel', 'X_legs_ang_vel',
                  'Y_arms_ang_vel', 'Y_torso_ang_vel', 'Y_legs_ang_vel']
    
    for col in quant_cols:
        if col in rally_frames.columns:
            vals = rally_frames[col].dropna()
            if len(vals) > 0:
                feat_dict[f'{col}_mean'] = vals.mean()
                feat_dict[f'{col}_std'] = vals.std() if len(vals) > 1 else 0
                feat_dict[f'{col}_max'] = vals.max()
                feat_dict[f'{col}_min'] = vals.min()
                feat_dict[f'{col}_range'] = vals.max() - vals.min()
                # First and last values
                feat_dict[f'{col}_first'] = vals.iloc[0]
                feat_dict[f'{col}_last'] = vals.iloc[-1]
    
    # End state (landing)
    end_row = rally_frames.iloc[-1]
    feat_dict['shuttle_x_end'] = end_row.get('shuttle_x', 0.5)
    feat_dict['shuttle_y_end'] = end_row.get('shuttle_y', 0.5)
    
    # Landing zone
    shuttle_y_end = feat_dict['shuttle_y_end']
    shuttle_x_end = feat_dict['shuttle_x_end']
    feat_dict['dist_from_center'] = abs(shuttle_x_end - 0.5)
    feat_dict['is_deep'] = 1 if shuttle_y_end > 0.7 else 0
    feat_dict['is_corner'] = 1 if (abs(shuttle_x_end - 0.5) > 0.15 and shuttle_y_end > 0.6) else 0
    
    # Motion from rally labels
    feat_dict['avg_motion'] = row.get('avg_motion', 0)
    feat_dict['max_motion'] = row.get('max_motion', 0)
    feat_dict['frame_count'] = row.get('frame_count', 0)
    
    # Stance features
    for player in ['X', 'Y']:
        stance_col = f'{player}_stance_width'
        if stance_col in rally_frames_frame.columns:
            stance_vals = pd.to_numeric(rally_frames_frame[stance_col], errors='coerce').dropna()
            if len(stance_vals) > 0:
                feat_dict[f'{player}_stance_mean'] = stance_vals.mean()
                feat_dict[f'{player}_stance_std'] = stance_vals.std() if len(stance_vals) > 1 else 0
    
    features_list.append(feat_dict)

df = pd.DataFrame(features_list)
print(f"Built {len(df)} rally samples, {len([c for c in df.columns if c not in ['rally_idx', 'winner']])} features")

feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].values
y = df['winner'].values

print(f"Class distribution: winner=1: {sum(y==1)}, winner=0: {sum(y==0)}")

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Extensive RF param grid
param_grid = [
    {'n_estimators': 50, 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 50, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 4},
    {'n_estimators': 150, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 150, 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 3},
    {'n_estimators': 200, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 200, 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 4},
    {'n_estimators': 300, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 300, 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 3},
    {'n_estimators': 50, 'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 5},
    {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 5},
    {'n_estimators': 100, 'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 4},
    {'n_estimators': 100, 'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 3},
    {'n_estimators': 150, 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 250, 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 250, 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 3},
    {'n_estimators': 500, 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 4},
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_params = None
results = []

for params in param_grid:
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            min_samples_split=params['min_samples_split'],
            random_state=42+fold,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        cv_scores.append(balanced_accuracy_score(y_test, pred))
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    results.append({'params': params, 'score': mean_score, 'std': std_score})
    
    if mean_score > best_score:
        best_score = mean_score
        best_params = params
        print(f"New best: {params} -> CV: {mean_score:.3f}")

print(f"\nBest: {best_params} -> CV: {best_score:.3f}")
print(f"Previous best (v19/v27/v28): CV 0.733")

# Final model
final_rf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42,
    class_weight='balanced'
)
final_rf.fit(X, y)

# Feature importance
importances = dict(zip(feature_cols, final_rf.feature_importances_))
top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])
print(f"Top features: {list(top_features.keys())}")

# Temporal test
test_size = 8
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

model_test = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42,
    class_weight='balanced'
)
model_test.fit(X_train, y_train)
y_pred = model_test.predict(X_test)

temporal_acc = accuracy_score(y_test, y_pred)
temporal_bal = balanced_accuracy_score(y_test, y_pred)

# Save report
report = {
    'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
    'step_executed': 'Train v30 - All features + extensive RF search',
    'samples': len(df),
    'train': len(df) - test_size,
    'test': test_size,
    'features': len(feature_cols),
    'feature_names': feature_cols,
    'class_distribution': {'winner_1': int(sum(y==1)), 'winner_0': int(sum(y==0))},
    'best_params': best_params,
    'cv_balanced_accuracy': round(best_score, 3),
    'all_results_count': len(results),
    'temporal_test': {
        'accuracy': round(temporal_acc, 3),
        'balanced_accuracy': round(temporal_bal, 3)
    },
    'top_features': {k: round(float(v), 4) for k, v in top_features.items()},
    'comparison_to_v19': 'v19 CV 0.733, v30 CV ' + str(round(best_score, 3)),
    'next_step': 'Plateau likely due to limited data (42 samples). Consider data augmentation.'
}

out_path = ROOT / 'reports' / f"train_v30_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"Saved: {out_path}")
