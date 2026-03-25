#!/usr/bin/env python3
"""Train v27: Focus on shuttle landing position + trajectory features"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Load quant features (v11 has angular velocity)
with gzip.open(ROOT / 'data/quant_features_v11.csv.gz', 'rt') as f:
    quant = pd.read_csv(f)
print(f"Loaded {len(quant)} frame quant features")

# Load frame features for shuttle positions and stance
import json as json_lib
frame_features = []
with open(ROOT / 'data/frame_features_v10.jsonl', 'r') as f:
    for line in f:
        frame_features.append(json_lib.loads(line))
frame_df = pd.DataFrame(frame_features)
print(f"Loaded {len(frame_df)} frame features")

# Build rally-level features with emphasis on shuttle landing
features_list = []
for _, row in rally_v9.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    rally_frames = quant[(quant['frame'] >= start) & (quant['frame'] <= end)]
    rally_frames_frame = frame_df[(frame_df['frame'] >= start) & (frame_df['frame'] <= end)]
    
    if len(rally_frames) < 3:
        continue
    
    # Get end state (landing position)
    end_row = rally_frames.iloc[-1]
    shuttle_x_end = end_row['shuttle_x']
    shuttle_y_end = end_row['shuttle_y']
    
    # Compute landing zone features
    # Court zones: front (y<0.4), mid (0.4-0.7), deep (y>0.7)
    # Left (x<0.4), center (0.4-0.6), right (x>0.6)
    zone_y = 0  # front
    if shuttle_y_end > 0.7:
        zone_y = 2  # deep
    elif shuttle_y_end > 0.4:
        zone_y = 1  # mid
    
    zone_x = 1  # center
    if shuttle_x_end < 0.4:
        zone_x = 0  # left
    elif shuttle_x_end > 0.6:
        zone_x = 2  # right
    
    # Distance from center line
    dist_from_center = abs(shuttle_x_end - 0.5)
    is_deep = 1 if shuttle_y_end > 0.7 else 0
    is_corner = 1 if (dist_from_center > 0.15 and shuttle_y_end > 0.6) else 0
    
    # Shuttle trajectory features
    shuttle_speeds = rally_frames['shuttle_speed'].dropna()
    shuttle_speed_max = shuttle_speeds.max() if len(shuttle_speeds) > 0 else 0
    shuttle_speed_mean = shuttle_speeds.mean() if len(shuttle_speeds) > 0 else 0
    
    # Shuttle direction changes
    dir_changes = rally_frames['shuttle_dir_change'].dropna()
    dir_change_count = (dir_changes > 0).sum() if len(dir_changes) > 0 else 0
    dir_change_sum = dir_changes.sum() if len(dir_changes) > 0 else 0
    
    feat_dict = {
        'rally_idx': len(features_list),
        'winner': int(row['winner']),
        'shuttle_x_end': shuttle_x_end,
        'shuttle_y_end': shuttle_y_end,
        'dist_from_center': dist_from_center,
        'is_deep': is_deep,
        'is_corner': is_corner,
        'zone_y': zone_y,
        'zone_x': zone_x,
        'shuttle_speed_max': shuttle_speed_max,
        'shuttle_speed_mean': shuttle_speed_mean,
        'dir_change_count': dir_change_count,
        'dir_change_sum': dir_change_sum,
    }
    
    # Angular velocity features (from quant v11)
    for player in ['X', 'Y']:
        for joint in ['arms', 'torso', 'legs']:
            col = f"{player}_{joint}_ang_vel"
            if col in rally_frames.columns:
                vals = rally_frames[col].dropna()
                feat_dict[f'{player}_{joint}_ang_vel_max'] = vals.max() if len(vals) else 0
                feat_dict[f'{player}_{joint}_ang_vel_mean'] = vals.mean() if len(vals) else 0
    
    # Motion features from rally labels
    feat_dict['avg_motion'] = row.get('avg_motion', 0)
    feat_dict['max_motion'] = row.get('max_motion', 0)
    
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
print(f"Built {len(df)} rally samples")
print(f"Features: {len([c for c in df.columns if c not in ['rally_idx', 'winner']])}")

feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].values
y = df['winner'].values

print(f"Class distribution: winner=1: {sum(y==1)}, winner=0: {sum(y==0)}")

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

selector = SelectKBest(f_classif, k=min(12, len(feature_cols)))
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected features: {selected_features}")

# Extended param grid
param_grid = [
    {'n_estimators': 50, 'max_depth': 3, 'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 3, 'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 1},
    {'n_estimators': 150, 'max_depth': 4, 'min_samples_leaf': 3},
    {'n_estimators': 150, 'max_depth': 5, 'min_samples_leaf': 2},
    {'n_estimators': 200, 'max_depth': 3, 'min_samples_leaf': 2},
    {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 1},
    {'n_estimators': 50, 'max_depth': None, 'min_samples_leaf': 5},
    {'n_estimators': 100, 'max_depth': 6, 'min_samples_leaf': 2},
    {'n_estimators': 300, 'max_depth': 4, 'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 2, 'min_samples_leaf': 1},
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_params = None
results = []

for params in param_grid:
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=42+fold,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        cv_scores.append(balanced_accuracy_score(y_test, pred))
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    results.append({'params': params, 'score': mean_score, 'std': std_score})
    print(f"Params: {params} -> CV: {mean_score:.3f} (+/- {std_score:.3f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_params = params

print(f"\nBest: {best_params} -> CV: {best_score:.3f}")
print(f"Previous best (v19): CV 0.733")

# Train final model
final_rf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params.get('min_samples_leaf', 1),
    random_state=42,
    class_weight='balanced'
)
final_rf.fit(X_selected, y)

# Temporal test
test_size = 8
X_train, X_test = X_selected[:-test_size], X_selected[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

model_test = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params.get('min_samples_leaf', 1),
    random_state=42,
    class_weight='balanced'
)
model_test.fit(X_train, y_train)
y_pred = model_test.predict(X_test)

temporal_acc = accuracy_score(y_test, y_pred)
temporal_bal = balanced_accuracy_score(y_test, y_pred)

importances = dict(zip(selected_features, final_rf.feature_importances_))
top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:8])
print(f"Top features: {top_features}")

# Save report
report = {
    'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
    'step_executed': 'Train v27 - Shuttle landing zone + trajectory features',
    'samples': len(df),
    'train': len(df) - test_size,
    'test': test_size,
    'features': len(selected_features),
    'feature_names': selected_features,
    'class_distribution': {'winner_1': int(sum(y==1)), 'winner_0': int(sum(y==0))},
    'best_params': best_params,
    'cv_balanced_accuracy': round(best_score, 3),
    'all_results': [{'params': str(r['params']), 'score': round(r['score'], 3), 'std': round(r['std'], 3)} for r in results],
    'temporal_test': {
        'accuracy': round(temporal_acc, 3),
        'balanced_accuracy': round(temporal_bal, 3)
    },
    'top_features': {k: round(float(v), 3) for k, v in top_features.items()},
    'comparison_to_v19': 'v19 CV 0.733, v27 CV ' + str(round(best_score, 3)),
    'next_step': 'Try GradientBoosting or collect more training data'
}

out_path = ROOT / 'reports' / f"train_v27_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"Saved: {out_path}")
