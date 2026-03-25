#!/usr/bin/env python3
"""
train_v32.py - Use balanced rally labels (v9) + angular velocity features (v11) + class_weight
Heartbeat step: Combine best labels (v9 - balanced 28:14) with best features (v11 - angular velocity)
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
print("Loading v11 features (with angular velocity)...")
with gzip.open(f"{ROOT}/data/quant_features_v11.csv.gz", 'rt') as f:
    df = pd.read_csv(f)

# Load v9 labels (balanced 28:14)
print("Loading v9 labels (balanced 28:14)...")
df_labels = pd.read_csv(f"{ROOT}/data/rally_labels_v9.csv.gz")

print(f"Features shape: {df.shape}")
print(f"Labels: {len(df_labels)}")
print(f"Class distribution: {dict(df_labels['winner'].value_counts())}")

# Build rally-level features (using angular velocity from v11)
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
            'rally_id': idx,  # Use index as rally_id
            'winner': row['winner'],
            'start_frame': sf,
            'end_frame': ef,
            # Basic shuttle features
            'shuttle_y_end': rally_df['shuttle_y'].iloc[-1] if 'shuttle_y' in rally_df else 0,
            'shuttle_x_end': rally_df['shuttle_x'].iloc[-1] if 'shuttle_x' in rally_df else 0,
            'is_deep': 1 if rally_df['shuttle_y'].iloc[-1] > 0.5 else 0,
            'is_corner': 1 if (rally_df['shuttle_x'].iloc[-1] < 0.2 or rally_df['shuttle_x'].iloc[-1] > 0.8) else 0,
            # Angular velocity features (from v11)
            'X_arms_ang_vel_max': rally_df['X_arms_ang_vel'].max() if 'X_arms_ang_vel' in rally_df else 0,
            'X_arms_ang_vel_mean': rally_df['X_arms_ang_vel'].mean() if 'X_arms_ang_vel' in rally_df else 0,
            'X_torso_ang_vel_max': rally_df['X_torso_ang_vel'].max() if 'X_torso_ang_vel' in rally_df else 0,
            'X_torso_ang_vel_mean': rally_df['X_torso_ang_vel'].mean() if 'X_torso_ang_vel' in rally_df else 0,
            'X_legs_ang_vel_max': rally_df['X_legs_ang_vel'].max() if 'X_legs_ang_vel' in rally_df else 0,
            'X_legs_ang_vel_mean': rally_df['X_legs_ang_vel'].mean() if 'X_legs_ang_vel' in rally_df else 0,
            'Y_arms_ang_vel_max': rally_df['Y_arms_ang_vel'].max() if 'Y_arms_ang_vel' in rally_df else 0,
            'Y_arms_ang_vel_mean': rally_df['Y_arms_ang_vel'].mean() if 'Y_arms_ang_vel' in rally_df else 0,
            # Motion features
            'avg_motion': rally_df['shuttle_speed'].mean() if 'shuttle_speed' in rally_df else 0,
            'max_motion': rally_df['shuttle_speed'].max() if 'shuttle_speed' in rally_df else 0,
            'min_motion': rally_df['shuttle_speed'].min() if 'shuttle_speed' in rally_df else 0,
            # Stance features
            'X_stance_mean': rally_df['X_torso_rot'].mean() if 'X_torso_rot' in rally_df else 0,
            'X_stance_std': rally_df['X_torso_rot'].std() if 'X_torso_rot' in rally_df else 0,
            'Y_stance_mean': rally_df['Y_torso_rot'].mean() if 'Y_torso_rot' in rally_df else 0,
            'Y_stance_std': rally_df['Y_torso_rot'].std() if 'Y_torso_rot' in rally_df else 0,
            # Rally duration
            'rally_duration': ef - sf,
        }
        features.append(feat)
    return pd.DataFrame(features)

df_feat = build_rally_features(df, df_labels)
print(f"\nRallies with features: {len(df_feat)}")
print(f"Class distribution: {dict(df_feat['winner'].value_counts())}")

feature_cols = [
    'shuttle_y_end', 'shuttle_x_end', 'is_deep', 'is_corner',
    'X_arms_ang_vel_max', 'X_arms_ang_vel_mean',
    'X_torso_ang_vel_max', 'X_torso_ang_vel_mean',
    'X_legs_ang_vel_max', 'X_legs_ang_vel_mean',
    'Y_arms_ang_vel_max', 'Y_arms_ang_vel_mean',
    'avg_motion', 'max_motion', 'min_motion',
    'X_stance_mean', 'X_stance_std', 'Y_stance_mean', 'Y_stance_std',
    'rally_duration'
]

X = df_feat[feature_cols].values
y = df_feat['winner'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validate with class_weight='balanced'
print("\n=== Cross-Validation with class_weight='balanced' ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for n_est in [50, 100, 200]:
    for depth in [3, 5, 7, None]:
        for min_leaf in [1, 2, 3]:
            clf = RandomForestClassifier(
                n_estimators=n_est, 
                max_depth=depth,
                min_samples_leaf=min_leaf,
                class_weight='balanced', 
                random_state=42
            )
            scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='balanced_accuracy')
            results.append({
                'n_estimators': n_est,
                'max_depth': depth,
                'min_samples_leaf': min_leaf,
                'cv_mean': scores.mean(),
                'cv_std': scores.std()
            })

# Sort by CV mean
results_sorted = sorted(results, key=lambda x: x['cv_mean'], reverse=True)
print("\nTop 5 configurations:")
for r in results_sorted[:5]:
    print(f"n_est={r['n_estimators']}, depth={r['max_depth']}, min_leaf={r['min_samples_leaf']}: CV={r['cv_mean']:.3f}+/-{r['cv_std']:.3f}")

best = results_sorted[0]
print(f"\nBest: CV={best['cv_mean']:.3f}")

# Try without class_weight too
print("\n=== Without class_weight (baseline) ===")
results_no_weight = []
for n_est in [50, 100, 200]:
    for depth in [3, 5, 7, None]:
        for min_leaf in [1, 2, 3]:
            clf = RandomForestClassifier(
                n_estimators=n_est, 
                max_depth=depth,
                min_samples_leaf=min_leaf,
                random_state=42
            )
            scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='balanced_accuracy')
            results_no_weight.append({
                'n_estimators': n_est,
                'max_depth': depth,
                'min_samples_leaf': min_leaf,
                'cv_mean': scores.mean(),
                'cv_std': scores.std()
            })

results_no_weight_sorted = sorted(results_no_weight, key=lambda x: x['cv_mean'], reverse=True)
print("\nTop 5 without class_weight:")
for r in results_no_weight_sorted[:5]:
    print(f"n_est={r['n_estimators']}, depth={r['max_depth']}, min_leaf={r['min_samples_leaf']}: CV={r['cv_mean']:.3f}+/-{r['cv_std']:.3f}")

best_no_weight = results_no_weight_sorted[0]
print(f"\nBest without weight: CV={best_no_weight['cv_mean']:.3f}")

# Save report
report = {
    "timestamp_local": "2026-03-24T11-12-00-05:00",
    "step_executed": "Use v9 labels (balanced 28:14) + v11 features (angular velocity) + class_weight",
    "samples": len(df_feat),
    "class_distribution": {"winner_0": int((y==0).sum()), "winner_1": int((y==1).sum())},
    "best_cv_with_weight": float(best['cv_mean']),
    "best_cv_without_weight": float(best_no_weight['cv_mean']),
    "baseline_v19_cv": 0.733,
    "previous_v31_cv": 0.900,
    "best_config_with_weight": {
        "n_estimators": best['n_estimators'],
        "max_depth": best['max_depth'],
        "min_samples_leaf": best['min_samples_leaf']
    },
    "improvement": float(best['cv_mean']) - 0.733
}

with open(f"{ROOT}/reports/train_v32_2026-03-24T11-12-00.json", 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n=== Summary ===")
print(f"v19 baseline: CV=0.733")
print(f"v31 (v4 labels + class_weight): CV=0.900")
print(f"v32 (v9 labels + v11 features + class_weight): CV={best['cv_mean']:.3f}")
print(f"Improvement: {best['cv_mean'] - 0.733:+.3f}")
print("Report saved.")