#!/usr/bin/env python3
"""Train v25: Add relative positioning + class weighting"""
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

# Load quant features
with gzip.open(ROOT / 'data/quant_features_v11.csv.gz', 'rt') as f:
    quant = pd.read_csv(f)
print(f"Loaded {len(quant)} quant features")

# Load frame features for stance
frame_features = []
with open(ROOT / 'data/frame_features_v10.jsonl', 'r') as f:
    for line in f:
        frame_features.append(json.loads(line))
frame_df = pd.DataFrame(frame_features)
print(f"Loaded {len(frame_df)} frame features")

# Build rally-level features with relative positioning
features_list = []
for _, row in rally_v9.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    rally_frames = quant[(quant['frame'] >= start) & (quant['frame'] <= end)]
    rally_frames_frame = frame_df[(frame_df['frame'] >= start) & (frame_df['frame'] <= end)]
    
    if len(rally_frames) < 3:
        continue
    
    end_row = rally_frames.iloc[-1]
    shuttle_x, shuttle_y = end_row['shuttle_x'], end_row['shuttle_y']
    
    # Basic shot features
    feat_dict = {
        'rally_idx': len(features_list),
        'winner': int(row['winner']),
        'shuttle_x': shuttle_x,
        'shuttle_y': shuttle_y,
        'dist_from_center': abs(shuttle_x - 0.5),
        'is_deep': 1 if shuttle_y > 0.7 else 0,
        'is_wide': 1 if shuttle_x < 0.2 or shuttle_x > 0.8 else 0,
    }
    
    # Player positions at end of rally
    for player in ['X', 'Y']:
        px_col = f'{player}_x'
        py_col = f'{player}_y'
        if px_col in rally_frames.columns:
            feat_dict[f'{player}_pos_x'] = rally_frames[px_col].iloc[-1] if len(rally_frames) else 0.5
            feat_dict[f'{player}_pos_y'] = rally_frames[py_col].iloc[-1] if len(rally_frames) else 0.5
    
    # Relative positioning (NEW)
    if 'X_x' in rally_frames.columns and 'Y_x' in rally_frames.columns:
        feat_dict['player_dist_x'] = abs(rally_frames['X_x'].iloc[-1] - rally_frames['Y_x'].iloc[-1]) if len(rally_frames) else 0
        feat_dict['player_dist_y'] = abs(rally_frames['X_y'].iloc[-1] - rally_frames['Y_y'].iloc[-1]) if len(rally_frames) else 0
        feat_dict['player_dist_total'] = np.sqrt(feat_dict['player_dist_x']**2 + feat_dict['player_dist_y']**2)
        
        # Who is closer to shuttle
        shuttle_x_end, shuttle_y_end = shuttle_x, shuttle_y
        x_dist = np.sqrt((rally_frames['X_x'].iloc[-1] - shuttle_x_end)**2 + (rally_frames['X_y'].iloc[-1] - shuttle_y_end)**2)
        y_dist = np.sqrt((rally_frames['Y_x'].iloc[-1] - shuttle_x_end)**2 + (rally_frames['Y_y'].iloc[-1] - shuttle_y_end)**2)
        feat_dict['X_closer_to_shuttle'] = 1 if x_dist < y_dist else 0
    
    # Angular velocity features
    for player in ['X', 'Y']:
        for joint in ['arms', 'torso', 'legs']:
            col = f"{player}_{joint}_ang_vel"
            if col in rally_frames.columns:
                vals = rally_frames[col].dropna()
                feat_dict[f'{player}_{joint}_ang_vel_max'] = vals.max() if len(vals) else 0
                feat_dict[f'{player}_{joint}_ang_vel_mean'] = vals.mean() if len(vals) else 0
                feat_dict[f'{player}_{joint}_ang_vel_std'] = vals.std() if len(vals) else 0
    
    # Motion features from labels
    feat_dict['avg_motion'] = row.get('avg_motion', 0)
    feat_dict['max_motion'] = row.get('max_motion', 0)
    feat_dict['frame_count'] = row.get('frame_count', end - start)
    
    # Stance features
    for player in ['X', 'Y']:
        stance_col = f'{player}_stance_width'
        if stance_col in rally_frames_frame.columns:
            stance_vals = pd.to_numeric(rally_frames_frame[stance_col], errors='coerce').dropna()
            if len(stance_vals) > 0:
                feat_dict[f'{player}_stance_mean'] = stance_vals.mean()
                feat_dict[f'{player}_stance_std'] = stance_vals.std()
    
    features_list.append(feat_dict)

df = pd.DataFrame(features_list)
print(f"Built {len(df)} rally features")

# Prepare features
feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].copy()
y = df['winner'].values

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=feature_cols)

print(f"Features: {len(feature_cols)}, Samples: {len(y)}")
print(f"Class distribution: {np.bincount(y)}")

# Try multiple approaches
results = []

# Approach 1: RF with class_weight='balanced'
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, 
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    cv_scores.append(balanced_accuracy_score(y_val, pred))

results.append(('RF_balanced', np.mean(cv_scores)))
print(f"RF (balanced): CV = {np.mean(cv_scores):.3f}")

# Approach 2: GradientBoosting with more trees
cv_scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                                     random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    cv_scores.append(balanced_accuracy_score(y_val, pred))

results.append(('GB_tuned', np.mean(cv_scores)))
print(f"GradientBoosting (tuned): CV = {np.mean(cv_scores):.3f}")

# Approach 3: RF with feature selection
selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Top features: {selected_features[:5]}")

cv_scores = []
for train_idx, val_idx in skf.split(X_selected, y):
    X_train, X_val = X_selected[train_idx], X_selected[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    cv_scores.append(balanced_accuracy_score(y_val, pred))

results.append(('RF_selected', np.mean(cv_scores)))
print(f"RF (selected features): CV = {np.mean(cv_scores):.3f}")

# Find best
best_name, best_score = max(results, key=lambda x: x[1])
print(f"\nBest: {best_name} with CV = {best_score:.3f}")
print(f"v19 reference: CV = 0.733")

# Save model
best_clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
best_clf.fit(X, y)

import pickle
model_path = ROOT / 'models/v25_balanced.pkl'
model_path.parent.mkdir(exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump({'model': best_clf, 'features': feature_cols, 'imputer': imputer}, f)
print(f"Saved model to {model_path}")
