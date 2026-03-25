#!/usr/bin/env python3
"""Train v26: Replicate v19 with extended hyperparameter search"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
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

# Build rally-level features (EXACT same as v19)
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
    
    feat_dict = {
        'rally_idx': len(features_list),
        'winner': int(row['winner']),
        'shuttle_x_end': shuttle_x_end,
        'shuttle_y_end': shuttle_y_end,
        'dist_from_center': dist_from_center,
        'is_deep': is_deep,
    }
    
    for player in ['X', 'Y']:
        for joint in ['arms', 'torso', 'legs']:
            col = f"{player}_{joint}_ang_vel"
            if col in rally_frames.columns:
                vals = rally_frames[col].dropna()
                feat_dict[f'{player}_{joint}_ang_vel_max'] = vals.max() if len(vals) else 0
    
    feat_dict['avg_motion'] = row.get('avg_motion', 0)
    feat_dict['max_motion'] = row.get('max_motion', 0)
    
    for player in ['X', 'Y']:
        stance_col = f'{player}_stance_width'
        if stance_col in rally_frames_frame.columns:
            stance_vals = pd.to_numeric(rally_frames_frame[stance_col], errors='coerce').dropna()
            if len(stance_vals) > 0:
                feat_dict[f'{player}_stance_mean'] = stance_vals.mean()
    
    features_list.append(feat_dict)

df = pd.DataFrame(features_list)
print(f"Built {len(df)} rally samples with features")

feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].values
y = df['winner'].values

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Extended hyperparameter grid search (more than v19)
param_grid = [
    {'n_estimators': 50, 'max_depth': 3, 'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 1},
    {'n_estimators': 150, 'max_depth': 4, 'min_samples_leaf': 3},
    {'n_estimators': 200, 'max_depth': 3, 'min_samples_leaf': 2},
    {'n_estimators': 50, 'max_depth': None, 'min_samples_leaf': 5},
    {'n_estimators': 100, 'max_depth': 6, 'min_samples_leaf': 2},
    {'n_estimators': 150, 'max_depth': 5, 'min_samples_leaf': 2},
    {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 2},
    {'n_estimators': 250, 'max_depth': 3, 'min_samples_leaf': 1},
    {'n_estimators': 300, 'max_depth': 2, 'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 7, 'min_samples_leaf': 1},
    {'n_estimators': 150, 'max_depth': None, 'min_samples_leaf': 3},
]

best_score = 0
best_params = None
results = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Try different k values
for k in [8, 10, 12]:
    selector = SelectKBest(f_classif, k=min(k, len(feature_cols)))
    X_selected = selector.fit_transform(X, y)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    
    for params in param_grid:
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            cv_scores.append(balanced_accuracy_score(y_test, pred))
        
        mean_cv = np.mean(cv_scores)
        results.append((k, params, mean_cv))
        
        if mean_cv > best_score:
            best_score = mean_cv
            best_params = (k, params)
            print(f"New best: k={k}, {params}, CV={mean_cv:.3f}")

print(f"\nBest: k={best_params[0]}, {best_params[1]}, CV={best_score:.3f}")
print(f"v19 reference: CV = 0.733")

# Final model with best params
k_best, params_best = best_params
selector = SelectKBest(f_classif, k=min(k_best, len(feature_cols)))
X_selected = selector.fit_transform(X, y)

final_clf = RandomForestClassifier(random_state=42, n_jobs=-1, **params_best)
final_clf.fit(X_selected, y)

# Save model
import pickle
model_path = ROOT / 'models/v26.pkl'
model_path.parent.mkdir(exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': final_clf, 
        'features': feature_cols, 
        'selected_features': selected_features,
        'imputer': imputer,
        'selector': selector,
        'k': k_best,
        'params': params_best
    }, f)
print(f"Saved model to {model_path}")
