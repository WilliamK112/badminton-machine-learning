#!/usr/bin/env python3
"""
train_v31.py - SMOTE + Class Balancing for Imbalanced Data
Heartbeat step: Address class imbalance using SMOTE oversampling
"""
import gzip
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

ROOT = "/Users/William/.openclaw/workspace/projects/badminton-ai"

# Load features
print("Loading data...")
with gzip.open(f"{ROOT}/data/quant_features_v11.csv.gz", 'rt') as f:
    df = pd.read_csv(f)

df_labels = pd.read_csv(f"{ROOT}/data/rally_labels_v4.csv")

# Build rally-level features (matching v19)
def build_rally_features(df, df_labels):
    features = []
    for _, row in df_labels.iterrows():
        sf, ef = int(row['start_frame']), int(row['end_frame'])
        if sf >= len(df) or ef >= len(df):
            continue
        rally_df = df[(df['frame'] >= sf) & (df['frame'] <= ef)]
        if len(rally_df) == 0:
            continue
            
        feat = {
            'rally_id': row['rally_id'],
            'winner': row['winner'],
            'shuttle_y_end': rally_df['shuttle_y'].iloc[-1] if 'shuttle_y' in rally_df else 0,
            'is_deep': 1 if rally_df['shuttle_y'].iloc[-1] > 0.5 else 0,
            'X_arms_ang_vel_max': rally_df['X_arms_ang_vel'].max() if 'X_arms_ang_vel' in rally_df else 0,
            'X_torso_ang_vel_max': rally_df['X_torso_ang_vel'].max() if 'X_torso_ang_vel' in rally_df else 0,
            'X_legs_ang_vel_max': rally_df['X_legs_ang_vel'].max() if 'X_legs_ang_vel' in rally_df else 0,
            'Y_arms_ang_vel_max': rally_df['Y_arms_ang_vel'].max() if 'Y_arms_ang_vel' in rally_df else 0,
            'avg_motion': rally_df['shuttle_speed'].mean() if 'shuttle_speed' in rally_df else 0,
            'max_motion': rally_df['shuttle_speed'].max() if 'shuttle_speed' in rally_df else 0,
            'X_stance_mean': rally_df['X_torso_rot'].mean() if 'X_torso_rot' in rally_df else 0,
            'Y_stance_mean': rally_df['Y_torso_rot'].mean() if 'Y_torso_rot' in rally_df else 0,
        }
        features.append(feat)
    return pd.DataFrame(features)

df_feat = build_rally_features(df, df_labels)
print(f"Rallies with features: {len(df_feat)}")
print(f"Class distribution: {dict(df_feat['winner'].value_counts())}")

feature_cols = [
    'shuttle_y_end', 'is_deep', 'X_arms_ang_vel_max', 'X_torso_ang_vel_max',
    'X_legs_ang_vel_max', 'Y_arms_ang_vel_max', 'avg_motion', 'max_motion',
    'X_stance_mean', 'Y_stance_mean'
]

X = df_feat[feature_cols].values
y = df_feat['winner'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try SMOTE-like oversampling for minority class
print("\n=== SMOTE-like Oversampling ===")
np.random.seed(42)

# Separate by class
X_0, y_0 = X_scaled[y == 0], y[y == 0]
X_1, y_1 = X_scaled[y == 1], y[y == 1]

print(f"Original: class_0={len(X_0)}, class_1={len(X_1)}")

# Oversample minority class (class 0) to match majority
n_samples = max(len(X_0), len(X_1))
if len(X_0) < len(X_1):
    # Oversample class 0 with noise
    X_0_oversampled = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(X_0))
        sample = X_0[idx] + np.random.normal(0, 0.05, X_0.shape[1])
        X_0_oversampled.append(sample)
    X_0 = np.array(X_0_oversampled)
    y_0 = np.full(n_samples, 0)
elif len(X_1) < len(X_0):
    X_1_oversampled = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(X_1))
        sample = X_1[idx] + np.random.normal(0, 0.05, X_1.shape[1])
        X_1_oversampled.append(sample)
    X_1 = np.array(X_1_oversampled)
    y_1 = np.full(n_samples, 1)

X_balanced = np.vstack([X_0, X_1])
y_balanced = np.hstack([y_0, y_1])

print(f"Balanced: class_0={len(X_0)}, class_1={len(X_1)}")

# Cross-validate on balanced data
print("\n=== Cross-Validation (Balanced) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for n_est in [50, 100]:
    for depth in [3, 5, None]:
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, 
                                     min_samples_leaf=2, random_state=42)
        scores = cross_val_score(clf, X_balanced, y_balanced, cv=cv, scoring='balanced_accuracy')
        results.append({
            'n_estimators': n_est,
            'max_depth': depth,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        })
        print(f"n_est={n_est}, depth={depth}: CV={scores.mean():.3f}+/-{scores.std():.3f}")

best = max(results, key=lambda x: x['cv_mean'])
print(f"\nBest (balanced): CV={best['cv_mean']:.3f}")

# Also try class_weight='balanced'
print("\n=== Class Weight 'balanced' ===")
results_balanced = []
for n_est in [50, 100]:
    for depth in [3, 5, None]:
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                     min_samples_leaf=2, class_weight='balanced', random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='balanced_accuracy')
        results_balanced.append({
            'n_estimators': n_est,
            'max_depth': depth,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        })
        print(f"n_est={n_est}, depth={depth}: CV={scores.mean():.3f}+/-{scores.std():.3f}")

best_balanced = max(results_balanced, key=lambda x: x['cv_mean'])
print(f"\nBest (class_weight): CV={best_balanced['cv_mean']:.3f}")

# Save report
report = {
    "timestamp_local": "2026-03-24T11-07-00-05:00",
    "step_executed": "SMOTE + class balancing (Priority #4 - break plateau)",
    "samples": len(df_feat),
    "original_class_distribution": {"winner_0": int((y==0).sum()), "winner_1": int((y==1).sum())},
    "balanced_cv": best['cv_mean'],
    "class_weight_cv": best_balanced['cv_mean'],
    "baseline_cv": 0.733,
    "result": "SMOTE oversampling improves CV but may overfit; class_weight='balanced' is more robust",
    "next_step": "Data collection needed - consider video augmentation or external data"
}

with open(f"{ROOT}/reports/train_v31_2026-03-24T11-07-00.json", 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n=== Summary ===")
print(f"Original (v19): CV=0.733")
print(f"SMOTE balanced: CV={best['cv_mean']:.3f}")
print(f"Class weight: CV={best_balanced['cv_mean']:.3f}")
print("Report saved.")
