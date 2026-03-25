#!/usr/bin/env python3
"""Train v18: Ensemble of RF + GB + ExtraTrees + LogReg (based on v17 best features)"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
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

# Build rally-level features (same as v17)
features_list = []
for _, row in rally_v9.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    rally_frames = quant[(quant['frame'] >= start) & (quant['frame'] <= end)]
    rally_frames_frame = frame_df[(frame_df['frame'] >= start) & (frame_df['frame'] <= end)]
    
    if len(rally_frames) < 3:
        continue
    
    # Endpoint features (from v13 - best performer)
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
    
    # Angular velocity max (key from v13)
    for player in ['X', 'Y']:
        for joint in ['arms', 'torso', 'legs']:
            col = f"{player}_{joint}_ang_vel"
            if col in rally_frames.columns:
                vals = rally_frames[col].dropna()
                feat_dict[f'{player}_{joint}_ang_vel_max'] = vals.max() if len(vals) else 0
    
    # Motion features (from v15)
    feat_dict['avg_motion'] = row.get('avg_motion', 0)
    feat_dict['max_motion'] = row.get('max_motion', 0)
    
    # Stance features (select best only)
    for player in ['X', 'Y']:
        stance_col = f'{player}_stance_width'
        if stance_col in rally_frames_frame.columns:
            stance_vals = pd.to_numeric(rally_frames_frame[stance_col], errors='coerce').dropna()
            if len(stance_vals) > 0:
                feat_dict[f'{player}_stance_mean'] = stance_vals.mean()
    
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

# Feature selection - select top 10 features (same as v17)
selector = SelectKBest(f_classif, k=min(10, len(feature_cols)))
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected features: {selected_features}")

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Class distribution: winner=1: {sum(y==1)}, winner=0: {sum(y==0)}")

# 5-fold stratified CV with ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_ensemble = []
cv_scores_rf = []
cv_scores_gb = []
cv_scores_et = []

# Scale features for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    X_train_scaled, X_test_scaled = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Individual models
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42+fold, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42+fold)
    et = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=42+fold, class_weight='balanced')
    lr = LogisticRegression(max_iter=500, random_state=42+fold, class_weight='balanced')
    
    # Fit individual models
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    et.fit(X_train, y_train)
    lr.fit(X_train_scaled, y_train)
    
    # Soft voting ensemble
    pred_rf = rf.predict_proba(X_test)[:, 1]
    pred_gb = gb.predict_proba(X_test)[:, 1]
    pred_et = et.predict_proba(X_test)[:, 1]
    pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Weighted average (higher weight for RF since it performed best in v17)
    ensemble_proba = 0.35 * pred_rf + 0.25 * pred_gb + 0.25 * pred_et + 0.15 * pred_lr
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    cv_scores_ensemble.append(balanced_accuracy_score(y_test, ensemble_pred))
    cv_scores_rf.append(balanced_accuracy_score(y_test, rf.predict(X_test)))
    cv_scores_gb.append(balanced_accuracy_score(y_test, gb.predict(X_test)))
    cv_scores_et.append(balanced_accuracy_score(y_test, et.predict(X_test)))

print(f"Ensemble CV balanced accuracy: {np.mean(cv_scores_ensemble):.3f} (+/- {np.std(cv_scores_ensemble):.3f})")
print(f"RF CV: {np.mean(cv_scores_rf):.3f}, GB CV: {np.mean(cv_scores_gb):.3f}, ET CV: {np.mean(cv_scores_et):.3f}")

# Use the best performing approach
scores = {
    'Ensemble': np.mean(cv_scores_ensemble),
    'RF': np.mean(cv_scores_rf),
    'GB': np.mean(cv_scores_gb),
    'ET': np.mean(cv_scores_et)
}
best_model_name = max(scores, key=scores.get)
print(f"Best: {best_model_name} = {scores[best_model_name]:.3f}")

# Temporal test (last 8 samples)
test_size = 8
X_train, X_test = X_selected[:-test_size], X_selected[-test_size:]
X_train_scaled, X_test_scaled = X_scaled[:-test_size], X_scaled[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# Train final models
rf_final = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight='balanced')
gb_final = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
et_final = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight='balanced')
lr_final = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')

rf_final.fit(X_train, y_train)
gb_final.fit(X_train, y_train)
et_final.fit(X_train, y_train)
lr_final.fit(X_train_scaled, y_train)

# Ensemble prediction
pred_rf = rf_final.predict_proba(X_test)[:, 1]
pred_gb = gb_final.predict_proba(X_test)[:, 1]
pred_et = et_final.predict_proba(X_test)[:, 1]
pred_lr = lr_final.predict_proba(X_test_scaled)[:, 1]

ensemble_proba = 0.35 * pred_rf + 0.25 * pred_gb + 0.25 * pred_et + 0.15 * pred_lr
y_pred = (ensemble_proba >= 0.5).astype(int)

temporal_acc = accuracy_score(y_test, y_pred)
temporal_bal = balanced_accuracy_score(y_test, y_pred)
temporal_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
temporal_mcc = matthews_corrcoef(y_test, y_pred)

print(f"Temporal test ({test_size} samples): acc={temporal_acc:.3f}, balanced={temporal_bal:.3f}")

# Feature importance (from RF)
importances = dict(zip(selected_features, rf_final.feature_importances_))
top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:8])
print(f"Top features: {top_features}")

# Save report
report = {
    'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
    'step_executed': 'Train v18 - Ensemble (RF+GB+ET+LR) with weighted voting',
    'samples': len(df),
    'train': len(df) - test_size,
    'test': test_size,
    'features': len(selected_features),
    'feature_names': selected_features,
    'class_distribution': {'winner_1': int(sum(y==1)), 'winner_0': int(sum(y==0))},
    'ensemble_weights': {'RF': 0.35, 'GB': 0.25, 'ET': 0.25, 'LR': 0.15},
    'cv_balanced_accuracy': {
        'ensemble': round(np.mean(cv_scores_ensemble), 3),
        'rf': round(np.mean(cv_scores_rf), 3),
        'gb': round(np.mean(cv_scores_gb), 3),
        'et': round(np.mean(cv_scores_et), 3),
    },
    'cv_std': round(np.std(cv_scores_ensemble), 3),
    'temporal_test': {
        'accuracy': round(temporal_acc, 3),
        'balanced_accuracy': round(temporal_bal, 3),
        'f1_macro': round(temporal_f1, 3),
        'mcc': round(temporal_mcc, 3)
    },
    'top_features': {k: round(float(v), 3) for k, v in top_features.items()},
    'next_step': 'Try more diverse features or collect more data'
}

out_path = ROOT / 'reports' / f"train_v18_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"Saved: {out_path}")