#!/usr/bin/env python3
"""Train v21: Try HistGradientBoosting + more features + different hyperparameters"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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

# Load frame features for stance
import json as json_lib
frame_features = []
with open(ROOT / 'data/frame_features_v10.jsonl', 'r') as f:
    for line in f:
        frame_features.append(json_lib.loads(line))
frame_df = pd.DataFrame(frame_features)
print(f"Loaded {len(frame_df)} frame features with stance data")

# Build rally-level features - expanded version
features_list = []
for _, row in rally_v9.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    rally_frames = quant[(quant['frame'] >= start) & (quant['frame'] <= end)]
    rally_frames_frame = frame_df[(frame_df['frame'] >= start) & (frame_df['frame'] <= end)]
    
    if len(rally_frames) < 3:
        continue
    
    # Shuttle end position features
    end_row = rally_frames.iloc[-1]
    shuttle_x_end = end_row['shuttle_x']
    shuttle_y_end = end_row['shuttle_y']
    start_row = rally_frames.iloc[0]
    shuttle_x_start = start_row['shuttle_x']
    shuttle_y_start = start_row['shuttle_y']
    
    # Shuttle trajectory
    shuttle_travel = np.sqrt((shuttle_x_end - shuttle_x_start)**2 + (shuttle_y_end - shuttle_y_start)**2)
    dist_from_center = abs(shuttle_x_end - 0.5)
    is_deep = 1 if shuttle_y_end > 0.7 else 0
    
    feat_dict = {
        'rally_idx': len(features_list),
        'winner': int(row['winner']),
        'shuttle_x_end': shuttle_x_end,
        'shuttle_y_end': shuttle_y_end,
        'shuttle_x_start': shuttle_x_start,
        'shuttle_y_start': shuttle_y_start,
        'shuttle_travel': shuttle_travel,
        'dist_from_center': dist_from_center,
        'is_deep': is_deep,
        'rally_duration': len(rally_frames),
    }
    
    # Motion features per player
    for player in ['X', 'Y']:
        for joint in ['arms', 'torso', 'legs']:
            col = f"{player}_{joint}_ang_vel"
            if col in rally_frames.columns:
                vals = rally_frames[col].dropna()
                if len(vals) > 0:
                    feat_dict[f'{player}_{joint}_ang_vel_max'] = vals.max()
                    feat_dict[f'{player}_{joint}_ang_vel_mean'] = vals.mean()
                    feat_dict[f'{player}_{joint}_ang_vel_std'] = vals.std() if len(vals) > 1 else 0
                    feat_dict[f'{player}_{joint}_ang_vel_min'] = vals.min()
                else:
                    feat_dict[f'{player}_{joint}_ang_vel_max'] = 0
                    feat_dict[f'{player}_{joint}_ang_vel_mean'] = 0
                    feat_dict[f'{player}_{joint}_ang_vel_std'] = 0
                    feat_dict[f'{player}_{joint}_ang_vel_min'] = 0
    
    # Overall motion
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
                feat_dict[f'{player}_stance_max'] = stance_vals.max()
    
    features_list.append(feat_dict)

df = pd.DataFrame(features_list)
print(f"Built {len(df)} rally samples with {len(df.columns)} features")

feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].values
y = df['winner'].values

# No imputation needed for HistGB, but keep for consistency
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

selector = SelectKBest(f_classif, k=min(15, len(feature_cols)))
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected features ({len(selected_features)}): {selected_features}")

print(f"Class distribution: winner=1: {sum(y==1)}, winner=0: {sum(y==0)}")

# Try HistGradientBoosting
print("\n=== HistGradientBoosting Training ===")
hgb_params = [
    {'max_iter': 50, 'max_depth': 3, 'learning_rate': 0.1, 'l2_regularization': 0.0},
    {'max_iter': 100, 'max_depth': 4, 'learning_rate': 0.1, 'l2_regularization': 0.0},
    {'max_iter': 100, 'max_depth': 5, 'learning_rate': 0.05, 'l2_regularization': 0.1},
    {'max_iter': 150, 'max_depth': 3, 'learning_rate': 0.1, 'l2_regularization': 0.1},
    {'max_iter': 200, 'max_depth': 4, 'learning_rate': 0.05, 'l2_regularization': 0.1},
    {'max_iter': 100, 'max_depth': None, 'learning_rate': 0.1, 'l2_regularization': 0.0},
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_params = None
hgb_results = []

for params in hgb_params:
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = HistGradientBoostingClassifier(
            max_iter=params['max_iter'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            l2_regularization=params['l2_regularization'],
            class_weight='balanced',
            random_state=42+fold
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        cv_scores.append(balanced_accuracy_score(y_test, pred))
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    hgb_results.append({'params': params, 'score': mean_score, 'std': std_score})
    print(f"HGB: {params} -> CV: {mean_score:.3f} (+/- {std_score:.3f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_params = params

print(f"\nBest HGB: {best_params} -> CV: {best_score:.3f}")

# Compare with RF (v19 approach) 
print("\n=== RandomForest (baseline comparison) ===")
rf_params = [
    {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 2},
    {'n_estimators': 150, 'max_depth': 5, 'min_samples_leaf': 1},
    {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 2},
]

rf_best_score = 0
rf_best_params = None
rf_results = []

for params in rf_params:
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            class_weight='balanced',
            random_state=42+fold
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        cv_scores.append(balanced_accuracy_score(y_test, pred))
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    rf_results.append({'params': params, 'score': mean_score, 'std': std_score})
    print(f"RF: {params} -> CV: {mean_score:.3f} (+/- {std_score:.3f})")
    
    if mean_score > rf_best_score:
        rf_best_score = mean_score
        rf_best_params = params

print(f"\nBest RF: {rf_best_params} -> CV: {rf_best_score:.3f}")

# Choose best model
if best_score >= rf_best_score:
    print(f"\n>>> HistGradientBoosting wins: {best_score:.3f} vs RF {rf_best_score:.3f}")
    use_hgb = True
    final_params = best_params
    final_score = best_score
    model_results = hgb_results
else:
    print(f"\n>>> RandomForest wins: {rf_best_score:.3f} vs HGB {best_score:.3f}")
    use_hgb = False
    final_params = rf_best_params
    final_score = rf_best_score
    model_results = rf_results

# Temporal test
test_size = 8
X_train, X_test = X_selected[:-test_size], X_selected[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

if use_hgb:
    final_model = HistGradientBoostingClassifier(
        max_iter=final_params['max_iter'],
        max_depth=final_params['max_depth'],
        learning_rate=final_params['learning_rate'],
        l2_regularization=final_params['l2_regularization'],
        class_weight='balanced',
        random_state=42
    )
else:
    final_model = RandomForestClassifier(
        n_estimators=final_params['n_estimators'],
        max_depth=final_params['max_depth'],
        min_samples_leaf=final_params['min_samples_leaf'],
        class_weight='balanced',
        random_state=42
    )

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

temporal_acc = accuracy_score(y_test, y_pred)
temporal_bal = balanced_accuracy_score(y_test, y_pred)
temporal_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
temporal_mcc = matthews_corrcoef(y_test, y_pred)

print(f"Temporal test ({test_size} samples): acc={temporal_acc:.3f}, balanced={temporal_bal:.3f}")

# Feature importances
if hasattr(final_model, 'feature_importances_'):
    importances = dict(zip(selected_features, final_model.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:8])
    print(f"Top features: {top_features}")
else:
    top_features = {}

# Save report
report = {
    'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
    'step_executed': 'Train v21 - HistGradientBoosting with expanded features',
    'samples': len(df),
    'total_features': len(feature_cols),
    'selected_features': len(selected_features),
    'feature_names': selected_features,
    'class_distribution': {'winner_1': int(sum(y==1)), 'winner_0': int(sum(y==0))},
    'best_model': 'HistGradientBoosting' if use_hgb else 'RandomForest',
    'best_params': final_params,
    'cv_balanced_accuracy': round(final_score, 3),
    'hgb_results': [{'params': str(r['params']), 'score': round(r['score'], 3)} for r in hgb_results],
    'rf_results': [{'params': str(r['params']), 'score': round(r['score'], 3)} for r in rf_results],
    'temporal_test': {
        'accuracy': round(temporal_acc, 3),
        'balanced_accuracy': round(temporal_bal, 3),
        'f1_macro': round(temporal_f1, 3),
        'mcc': round(temporal_mcc, 3)
    },
    'top_features': {k: round(float(v), 3) for k, v in top_features.items()},
    'comparison_to_v19': f'v19 CV=0.733, v21 CV={final_score:.3f}',
    'next_step': 'Try ensemble of best models or add more sophisticated features'
}

out_path = ROOT / 'reports' / f"train_v21_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\nSaved: {out_path}")