#!/usr/bin/env python3
"""
train_v35.py - Use v11 features (with angular velocity) + v9 labels (28:14 balanced)
Goal: Beat v31 CV=0.900 using all 42 rallies (not just 35)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Load v11 features (has angular velocity)
print("Loading v11 features (with angular velocity)...")
features_df = pd.read_csv('data/quant_features_v11.csv.gz')
print(f"  Features: {len(features_df)} frames, frame range {features_df['frame'].min()}-{features_df['frame'].max()}")

# Load v9 labels (28:14 balanced)
print("Loading v9 labels (28:14 balanced)...")
labels = pd.read_csv('data/rally_labels_v9.csv.gz')
print(f"  Labels: {len(labels)} rallies")
print(f"  Winner distribution: {labels['winner'].value_counts().to_dict()}")

# Feature columns - include angular velocity
feature_cols = [
    # Position features
    'shuttle_x', 'shuttle_y', 'shuttle_speed',
    'X_l_forearm', 'X_r_forearm', 'X_l_upperarm', 'X_r_upperarm',
    'X_torso_rot', 'X_l_thigh', 'X_r_thigh', 'X_l_calf', 'X_r_calf',
    'Y_l_forearm', 'Y_r_forearm', 'Y_l_upperarm', 'Y_r_upperarm',
    'Y_torso_rot', 'Y_l_thigh', 'Y_r_thigh', 'Y_l_calf', 'Y_r_calf',
    # Angular velocity (the key addition!)
    'X_arms_ang_vel', 'X_torso_ang_vel', 'X_legs_ang_vel',
]

# Filter to available columns
available_cols = [c for c in feature_cols if c in features_df.columns]
print(f"  Using {len(available_cols)} features: {available_cols[:5]}...")

# Build rally-level dataset
print("\nBuilding rally-level dataset...")
X_list = []
y_list = []
rally_info = []

for idx, row in labels.iterrows():
    start = int(row['start_frame'])
    end = int(row['end_frame'])
    
    # Get frames in this rally
    rally_frames = features_df[(features_df['frame'] >= start) & (features_df['frame'] <= end)]
    
    if len(rally_frames) == 0:
        print(f"  Warning: No features for rally {start}-{end}, skipping")
        continue
    
    # Aggregate features: mean, std, min, max for each column
    rally_features = {}
    for col in available_cols:
        vals = rally_frames[col].dropna()
        if len(vals) > 0:
            rally_features[f'{col}_mean'] = vals.mean()
            rally_features[f'{col}_std'] = vals.std() if len(vals) > 1 else 0
            rally_features[f'{col}_max'] = vals.max()
    
    X_list.append(rally_features)
    y_list.append(int(row['winner']))
    rally_info.append({'start': start, 'end': end, 'frames': len(rally_frames)})

X = pd.DataFrame(X_list).fillna(0)
y = np.array(y_list)

print(f"  Dataset: {len(X)} rallies, {X.shape[1]} features")
print(f"  Class distribution: {np.bincount(y)}")

# Train with cross-validation
print("\nTraining RandomForest with class_weight='balanced'...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=3,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

print(f"  CV Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
print(f"  Per-fold: {scores}")

# Train final model
clf.fit(X, y)
print(f"\nFinal model trained on {len(X)} samples")

# Save model
import pickle
with open('models/v35.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("Saved model to models/v35.pkl")

# Feature importance
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 important features:")
print(importances.head(10))

# Save feature list for prediction
with open('models/v35_features.json', 'w') as f:
    json.dump({'feature_cols': list(X.columns), 'label_cols': list(labels.columns)}, f)
print("Saved feature config to models/v35_features.json")

print(f"\n✓ v35 complete: CV={scores.mean():.3f}")