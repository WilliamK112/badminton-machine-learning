#!/usr/bin/env python3
"""
train_v36.py - Simpler features, proper CV with v9 labels (28:14 balanced)
Lesson: v35 had 72 features for 42 samples - too many! 
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

ROOT = "/Users/William/.openclaw/workspace/projects/badminton-ai"

# Load v11 features (has angular velocity)
print("Loading v11 features (with angular velocity)...")
features_df = pd.read_csv(f"{ROOT}/data/quant_features_v11.csv.gz")

# Load v9 labels (28:14 balanced)
print("Loading v9 labels (28:14 balanced)...")
labels = pd.read_csv(f"{ROOT}/data/rally_labels_v9.csv.gz")
print(f"  Labels: {len(labels)} rallies, winner distribution: {labels['winner'].value_counts().to_dict()}")

# Simpler feature set (matching v19 style but with angular velocity)
feature_cols = [
    'shuttle_x', 'shuttle_y', 'shuttle_speed',
    'X_torso_rot', 'X_arms_ang_vel', 'X_torso_ang_vel', 'X_legs_ang_vel',
    'Y_torso_rot', 'Y_arms_ang_vel', 'Y_torso_ang_vel', 'Y_legs_ang_vel',
]
available_cols = [c for c in feature_cols if c in features_df.columns]
print(f"  Using {len(available_cols)} features: {available_cols}")

# Build rally-level features - just mean values
print("\nBuilding rally-level dataset...")
X_list = []
y_list = []

for _, row in labels.iterrows():
    start = int(row['start_frame'])
    end = int(row['end_frame'])
    
    rally_frames = features_df[(features_df['frame'] >= start) & (features_df['frame'] <= end)]
    
    if len(rally_frames) == 0:
        continue
    
    # Simple aggregation: just mean
    rally_features = {}
    for col in available_cols:
        rally_features[col] = rally_frames[col].mean()
    
    X_list.append(rally_features)
    y_list.append(int(row['winner']))

X = pd.DataFrame(X_list).fillna(0)
y = np.array(y_list)

print(f"  Dataset: {len(X)} rallies, {X.shape[1]} features")
print(f"  Class distribution: {np.bincount(y)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validate multiple models
print("\n=== Testing Multiple Models ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

# RandomForest with class_weight
print("\nRandomForest (class_weight='balanced'):")
for n_est in [50, 100, 200]:
    for depth in [3, 5, None]:
        clf = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth,
            min_samples_leaf=2, class_weight='balanced', random_state=42
        )
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        results.append(('RF', n_est, depth, scores.mean(), scores.std()))
        print(f"  n_est={n_est}, depth={depth}: CV={scores.mean():.3f}+/-{scores.std():.3f}")

# GradientBoosting
print("\nGradientBoosting:")
for n_est in [50, 100]:
    for depth in [2, 3]:
        clf = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=0.1, random_state=42
        )
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        results.append(('GB', n_est, depth, scores.mean(), scores.std()))
        print(f"  n_est={n_est}, depth={depth}: CV={scores.mean():.3f}+/-{scores.std():.3f}")

# SVM
print("\nSVM (RBF):")
for C in [0.1, 1, 10]:
    clf = SVC(C=C, kernel='rbf', class_weight='balanced', random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    results.append(('SVM', C, 'rbf', scores.mean(), scores.std()))
    print(f"  C={C}: CV={scores.mean():.3f}+/-{scores.std():.3f}")

# Find best
best = max(results, key=lambda x: x[3])
print(f"\n✓ Best: {best[0]} with CV={best[3]:.3f}")

# Train final model with best params
print("\nTraining final model...")
if best[0] == 'RF':
    clf = RandomForestClassifier(
        n_estimators=best[1], max_depth=best[2],
        min_samples_leaf=2, class_weight='balanced', random_state=42
    )
elif best[0] == 'GB':
    clf = GradientBoostingClassifier(
        n_estimators=best[1], max_depth=best[2], learning_rate=0.1, random_state=42
    )
else:
    clf = SVC(C=best[1], kernel='rbf', class_weight='balanced', random_state=42)

clf.fit(X_scaled, y)

# Save model + scaler
with open('models/v36.pkl', 'wb') as f:
    pickle.dump({'model': clf, 'scaler': scaler, 'feature_cols': available_cols}, f)
print("Saved model to models/v36.pkl")

# Save results
report = {
    'timestamp_local': '2026-03-24T11-22-00-05:00',
    'step_executed': 'Simpler features (11) + multiple models with v9 labels (28:14)',
    'samples': len(X),
    'class_distribution': {'winner_0': int((y==0).sum()), 'winner_1': int((y==1).sum())},
    'best_model': best[0],
    'best_cv': best[3],
    'all_results': [(r[0], r[1], r[2], round(r[3], 3), round(r[4], 3)) for r in results],
}

with open('reports/train_v36_2026-03-24T11-22-00.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ v36 complete: Best CV={best[3]:.3f} ({best[0]})")
print(f"  Using v9 labels (28:14) vs v31 which used v4 (34:1) - more realistic evaluation")