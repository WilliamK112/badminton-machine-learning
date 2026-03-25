#!/usr/bin/env python3
"""
train_v33.py - Feature Augmentation for Minority Class
Heartbeat step: Augment minority class (winner_0) with feature noise/scaling to improve model robustness
"""
import gzip
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ROOT = "/Users/William/.openclaw/workspace/projects/badminton-ai"

# Load v11 features (with angular velocity)
print("Loading v11 features...")
with gzip.open(f"{ROOT}/data/quant_features_v11.csv.gz", 'rt') as f:
    df = pd.read_csv(f)

# Load v4 labels (used in v31)
df_labels = pd.read_csv(f"{ROOT}/data/rally_labels_v4.csv")

print(f"Features: {len(df)} frames")
print(f"Labels: {len(df_labels)} rallies")
print(f"Class distribution: {dict(df_labels['winner'].value_counts())}")

# Build rally-level features
def build_rally_features(df, df_labels):
    features = []
    for idx, row in df_labels.iterrows():
        sf, ef = int(row['start_frame']), int(row['end_frame'])
        if sf >= len(df) or ef >= len(df):
            continue
        rally_df = df[(df['frame'] >= sf) & (df['frame'] <= ef)]
        if len(rally_df) == 0:
            continue
        
        feat = {
            'winner': row['winner'],
            'shuttle_y_end': rally_df['shuttle_y'].iloc[-1] if 'shuttle_y' in rally_df else 0,
            'shuttle_x_end': rally_df['shuttle_x'].iloc[-1] if 'shuttle_x' in rally_df else 0,
            'is_deep': 1 if rally_df['shuttle_y'].iloc[-1] > 0.5 else 0,
            'is_corner': 1 if (rally_df['shuttle_x'].iloc[-1] < 0.2 or rally_df['shuttle_x'].iloc[-1] > 0.8) else 0,
            'X_arms_ang_vel_max': rally_df['X_arms_ang_vel'].max() if 'X_arms_ang_vel' in rally_df else 0,
            'X_torso_ang_vel_max': rally_df['X_torso_ang_vel'].max() if 'X_torso_ang_vel' in rally_df else 0,
            'X_legs_ang_vel_max': rally_df['X_legs_ang_vel'].max() if 'X_legs_ang_vel' in rally_df else 0,
            'Y_arms_ang_vel_max': rally_df['Y_arms_ang_vel'].max() if 'Y_arms_ang_vel' in rally_df else 0,
            'avg_motion': rally_df['shuttle_speed'].mean() if 'shuttle_speed' in rally_df else 0,
            'max_motion': rally_df['shuttle_speed'].max() if 'shuttle_speed' in rally_df else 0,
            'X_stance_mean': rally_df['X_torso_rot'].mean() if 'X_torso_rot' in rally_df else 0,
            'Y_stance_mean': rally_df['Y_torso_rot'].mean() if 'Y_torso_rot' in rally_df else 0,
            'rally_duration': ef - sf,
        }
        features.append(feat)
    return pd.DataFrame(features)

df_feat = build_rally_features(df, df_labels)
print(f"\nRallies with features: {len(df_feat)}")
print(f"Class distribution: {dict(df_feat['winner'].value_counts())}")

feature_cols = [
    'shuttle_y_end', 'shuttle_x_end', 'is_deep', 'is_corner',
    'X_arms_ang_vel_max', 'X_torso_ang_vel_max', 'X_legs_ang_vel_max', 'Y_arms_ang_vel_max',
    'avg_motion', 'max_motion', 'X_stance_mean', 'Y_stance_mean', 'rally_duration'
]

X = df_feat[feature_cols].values
y = df_feat['winner'].values

# Feature augmentation for minority class
print("\n=== Augmenting Minority Class ===")
X_0 = X[y == 0]  # Minority class (only 1 sample!)
X_1 = X[y == 1]  # Majority class

print(f"Original: class_0={len(X_0)}, class_1={len(X_1)}")

# Augment minority class with multiple transformations
np.random.seed(42)
augmented = []

# Create 20 augmented samples from the 1 minority sample
for i in range(20):
    for noise_scale in [0.05, 0.1, 0.15, 0.2]:
        # Add random noise
        noise = np.random.normal(0, noise_scale, X_0.shape[1])
        aug_sample = X_0[0] + noise
        augmented.append(aug_sample)

X_aug = np.array(augmented)
y_aug = np.zeros(len(X_aug))

print(f"Augmented minority samples: {len(X_aug)}")

# Combine original data with augmented minority class
X_combined = np.vstack([X, X_aug])
y_combined = np.hstack([y, y_aug])

print(f"Combined: class_0={len(X_combined[y_combined==0])}, class_1={len(X_combined[y_combined==1])}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Also prepare original data for comparison
X_orig_scaled = scaler.transform(X)

# Cross-validate on augmented data
print("\n=== Cross-Validation on Augmented Data ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_aug = []
for n_est in [50, 100]:
    for depth in [3, 5, None]:
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, 
                                     min_samples_leaf=2, random_state=42)
        scores = cross_val_score(clf, X_scaled, y_combined, cv=cv, scoring='balanced_accuracy')
        results_aug.append({
            'n_estimators': n_est,
            'max_depth': depth,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        })
        print(f"n_est={n_est}, depth={depth}: CV={scores.mean():.3f}+/-{scores.std():.3f}")

best_aug = max(results_aug, key=lambda x: x['cv_mean'])
print(f"\nBest (augmented): CV={best_aug['cv_mean']:.3f}")

# Also try class_weight on original data
print("\n=== Class Weight on Original Data ===")
results_cw = []
for n_est in [50, 100]:
    for depth in [3, 5, None]:
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                     min_samples_leaf=2, class_weight='balanced', random_state=42)
        scores = cross_val_score(clf, X_orig_scaled, y, cv=cv, scoring='balanced_accuracy')
        results_cw.append({
            'n_estimators': n_est,
            'max_depth': depth,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        })
        print(f"n_est={n_est}, depth={depth}: CV={scores.mean():.3f}+/-{scores.std():.3f}")

best_cw = max(results_cw, key=lambda x: x['cv_mean'])
print(f"\nBest (class_weight): CV={best_cw['cv_mean']:.3f}")

# Try both: augmented data + class_weight
print("\n=== Augmented + Class Weight ===")
results_both = []
for n_est in [50, 100]:
    for depth in [3, 5, None]:
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                     min_samples_leaf=2, class_weight='balanced', random_state=42)
        scores = cross_val_score(clf, X_scaled, y_combined, cv=cv, scoring='balanced_accuracy')
        results_both.append({
            'n_estimators': n_est,
            'max_depth': depth,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        })
        print(f"n_est={n_est}, depth={depth}: CV={scores.mean():.3f}+/-{scores.std():.3f}")

best_both = max(results_both, key=lambda x: x['cv_mean'])
print(f"\nBest (augmented + class_weight): CV={best_both['cv_mean']:.3f}")

# Save report
report = {
    "timestamp_local": "2026-03-24T11-12-00-05:00",
    "step_executed": "Feature augmentation for minority class",
    "original_samples": len(X),
    "augmented_samples": len(X_aug),
    "combined_samples": len(X_combined),
    "original_class_distribution": {"winner_0": int((y==0).sum()), "winner_1": int((y==1).sum())},
    "cv_augmented": float(best_aug['cv_mean']),
    "cv_class_weight": float(best_cw['cv_mean']),
    "cv_augmented_cw": float(best_both['cv_mean']),
    "baseline_v19_cv": 0.733,
    "previous_v31_cv": 0.900,
}

with open(f"{ROOT}/reports/train_v33_2026-03-24T11-12-00.json", 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n=== Summary ===")
print(f"v19 baseline: CV=0.733")
print(f"v31 (class_weight): CV=0.900")
print(f"v33 (augmented): CV={best_aug['cv_mean']:.3f}")
print(f"v33 (augmented + class_weight): CV={best_both['cv_mean']:.3f}")
print("Report saved.")