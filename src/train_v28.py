#!/usr/bin/env python3
"""Train v28: Try GradientBoosting + ExtraTrees + Voting ensemble"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]

# Load rally labels
with gzip.open(ROOT / 'data/rally_labels_v9.csv.gz', 'rt') as f:
    rally_v9 = pd.read_csv(f)
print(f"Loaded {len(rally_v9)} rallies")

# Load quant features
with gzip.open(ROOT / 'data/quant_features_v11.csv.gz', 'rt') as f:
    quant = pd.read_csv(f)

# Load frame features
import json as json_lib
frame_features = []
with open(ROOT / 'data/frame_features_v10.jsonl', 'r') as f:
    for line in f:
        frame_features.append(json_lib.loads(line))
frame_df = pd.DataFrame(frame_features)

# Build features (same as v27)
features_list = []
for _, row in rally_v9.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    rally_frames = quant[(quant['frame'] >= start) & (quant['frame'] <= end)]
    rally_frames_frame = frame_df[(frame_df['frame'] >= start) & (frame_df['frame'] <= end)]
    
    if len(rally_frames) < 3:
        continue
    
    end_row = rally_frames.iloc[-1]
    shuttle_x_end = end_row['shuttle_x']
    shuttle_y_end = end_row['shuttle_y']
    dist_from_center = abs(shuttle_x_end - 0.5)
    is_deep = 1 if shuttle_y_end > 0.7 else 0
    is_corner = 1 if (dist_from_center > 0.15 and shuttle_y_end > 0.6) else 0
    
    shuttle_speeds = rally_frames['shuttle_speed'].dropna()
    shuttle_speed_max = shuttle_speeds.max() if len(shuttle_speeds) > 0 else 0
    shuttle_speed_mean = shuttle_speeds.mean() if len(shuttle_speeds) > 0 else 0
    
    dir_changes = rally_frames['shuttle_dir_change'].dropna()
    dir_change_count = (dir_changes > 0).sum() if len(dir_changes) > 0 else 0
    
    feat_dict = {
        'rally_idx': len(features_list),
        'winner': int(row['winner']),
        'shuttle_x_end': shuttle_x_end,
        'shuttle_y_end': shuttle_y_end,
        'dist_from_center': dist_from_center,
        'is_deep': is_deep,
        'is_corner': is_corner,
        'shuttle_speed_max': shuttle_speed_max,
        'shuttle_speed_mean': shuttle_speed_mean,
        'dir_change_count': dir_change_count,
    }
    
    for player in ['X', 'Y']:
        for joint in ['arms', 'torso', 'legs']:
            col = f"{player}_{joint}_ang_vel"
            if col in rally_frames.columns:
                vals = rally_frames[col].dropna()
                feat_dict[f'{player}_{joint}_ang_vel_max'] = vals.max() if len(vals) else 0
                feat_dict[f'{player}_{joint}_ang_vel_mean'] = vals.mean() if len(vals) else 0
    
    feat_dict['avg_motion'] = row.get('avg_motion', 0)
    feat_dict['max_motion'] = row.get('max_motion', 0)
    
    for player in ['X', 'Y']:
        stance_col = f'{player}_stance_width'
        if stance_col in rally_frames_frame.columns:
            stance_vals = pd.to_numeric(rally_frames_frame[stance_col], errors='coerce').dropna()
            if len(stance_vals) > 0:
                feat_dict[f'{player}_stance_mean'] = stance_vals.mean()
                feat_dict[f'{player}_stance_std'] = stance_vals.std() if len(stance_vals) > 1 else 0
    
    features_list.append(feat_dict)

df = pd.DataFrame(features_list)
print(f"Built {len(df)} rally samples")

feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].values
y = df['winner'].values

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

selector = SelectKBest(f_classif, k=min(12, len(feature_cols)))
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected features: {selected_features}")

# Test multiple model types
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

# RF with best params from v27
rf_params = {'n_estimators': 50, 'max_depth': 3, 'min_samples_leaf': 2}
cv_scores = []
for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    rf = RandomForestClassifier(**rf_params, random_state=42+fold, class_weight='balanced')
    rf.fit(X_train, y_train)
    cv_scores.append(balanced_accuracy_score(y_test, rf.predict(X_test)))
print(f"RF: CV {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
results.append({'model': 'RF', 'cv': np.mean(cv_scores), 'std': np.std(cv_scores)})

# GradientBoosting
for n_est in [50, 100]:
    for max_d in [2, 3]:
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            gb = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_d, random_state=42+fold)
            gb.fit(X_train, y_train)
            cv_scores.append(balanced_accuracy_score(y_test, gb.predict(X_test)))
        print(f"GB(n={n_est}, d={max_d}): CV {np.mean(cv_scores):.3f}")
        results.append({'model': f'GB({n_est},{max_d})', 'cv': np.mean(cv_scores), 'std': np.std(cv_scores)})

# ExtraTrees
for n_est in [50, 100]:
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        et = ExtraTreesClassifier(n_estimators=n_est, max_depth=3, random_state=42+fold, class_weight='balanced')
        et.fit(X_train, y_train)
        cv_scores.append(balanced_accuracy_score(y_test, et.predict(X_test)))
    print(f"ET(n={n_est}): CV {np.mean(cv_scores):.3f}")
    results.append({'model': f'ET({n_est})', 'cv': np.mean(cv_scores), 'std': np.std(cv_scores)})

# Find best
best_result = max(results, key=lambda x: x['cv'])
print(f"\nBest: {best_result['model']} -> CV {best_result['cv']:.3f}")
print(f"Previous best (v19/v27): CV 0.733")

# Save report
report = {
    'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
    'step_executed': 'Train v28 - Model comparison (RF vs GB vs ET)',
    'samples': len(df),
    'features': len(selected_features),
    'feature_names': selected_features,
    'class_distribution': {'winner_1': int(sum(y==1)), 'winner_0': int(sum(y==0))},
    'model_comparison': results,
    'best_model': best_result['model'],
    'cv_balanced_accuracy': round(best_result['cv'], 3),
    'comparison_to_v19': 'v19 CV 0.733, v28 CV ' + str(round(best_result['cv'], 3)),
    'next_step': 'Consider collecting more training data or trying different feature engineering'
}

out_path = ROOT / 'reports' / f"train_v28_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"Saved: {out_path}")
