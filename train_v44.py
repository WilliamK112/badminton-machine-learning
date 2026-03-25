#!/usr/bin/env python3
"""
train_v44.py - Combine v11 features (angular velocity) + v13 features (landing/momentum)
Goal: Beat v41 CV=0.769
"""
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pickle
import warnings
warnings.filterwarnings('ignore')

ROOT = "/Users/William/.openclaw/workspace/projects/badminton-ai"

print("=" * 60)
print("TRAIN v44 - v11 + v13 Combined Features")
print("=" * 60)

# Load v11 features (has angular velocity)
print("\n[1] Loading v11 features...")
v11_df = pd.read_csv(f"{ROOT}/data/quant_features_v11.csv.gz")
print(f"    v11: {len(v11_df)} frames, {len(v11_df.columns)} cols")

# Load v13 frame features
print("[2] Loading v13 frame features...")
v13_frames = []
with open(f"{ROOT}/data/frame_features_v13.jsonl", "r") as f:
    for line in f:
        v13_frames.append(json.loads(line))
print(f"    v13: {len(v13_frames)} frames")

# Load v9 labels (balanced 28:14)
print("[3] Loading labels...")
labels = pd.read_csv(f"{ROOT}/data/rally_labels_v9.csv.gz")
print(f"    Labels: {len(labels)} rallies")

# Get v11 feature columns (excluding frame/t_sec/winner_proxy)
v11_feature_cols = [c for c in v11_df.columns if c not in ['frame', 't_sec', 'winner_proxy']]
print(f"    v11 feature cols: {len(v11_feature_cols)}")

# Build combined features
print("\n[4] Building combined features...")

def build_combined_features(labels_df, v11_df, v13_frames, v11_feature_cols):
    # Create v11 frame lookup (already has X_ and Y_ prefixed columns)
    v11_lookup = {row['frame']: row for _, row in v11_df.iterrows()}
    
    # Create v13 frame lookup
    v13_lookup = {f['frame']: f for f in v13_frames}
    
    X_list = []
    y_list = []
    
    for _, row in labels_df.iterrows():
        start = int(row['start_frame'])
        end = int(row['end_frame'])
        
        # Get v11 frames for this rally
        v11_rally = [v11_lookup.get(i) for i in range(start, end+1) if i in v11_lookup]
        v11_rally = [f for f in v11_rally if f is not None]
        
        # Get v13 frames
        v13_rally = [v13_lookup.get(i) for i in range(start, end+1) if i in v13_lookup]
        v13_rally = [f for f in v13_rally if f and f.get('players')]
        
        if len(v11_rally) == 0 and len(v13_rally) == 0:
            continue
        
        features = {}
        
        # === v11 features (angular velocity + basic) ===
        v11_df_rally = pd.DataFrame(v11_rally)
        if len(v11_df_rally) > 0:
            for col in v11_feature_cols:
                if col in v11_df_rally.columns:
                    features[f'{col}_mean'] = v11_df_rally[col].mean()
                    features[f'{col}_std'] = v11_df_rally[col].std()
                    features[f'{col}_max'] = v11_df_rally[col].max()
        
        # === v13 features (landing prediction, momentum, zone) ===
        if len(v13_rally) > 0:
            # Predicted landing
            pred_land_x = [f.get('predicted_landing_x', -1) for f in v13_rally if f.get('predicted_landing_x', -1) > 0]
            pred_land_y = [f.get('predicted_landing_y', -1) for f in v13_rally if f.get('predicted_landing_y', -1) > 0]
            if pred_land_x:
                features['pred_landing_x_mean'] = np.mean(pred_land_x)
                features['pred_landing_x_std'] = np.std(pred_land_x)
                features['pred_landing_y_mean'] = np.mean(pred_land_y)
                features['pred_landing_count'] = len(pred_land_x) / len(v13_rally)
            
            # Shuttle momentum
            momentum = [f.get('shuttle_momentum', 0) for f in v13_rally if f.get('shuttle_momentum') is not None]
            if momentum:
                features['momentum_mean'] = np.mean(momentum)
                features['momentum_std'] = np.std(momentum)
                features['momentum_max'] = np.max(np.abs(momentum))
            
            # Court zone distribution
            zones = [f.get('court_zone', 'net') for f in v13_rally]
            for zone in ['front_X', 'mid_X', 'net']:
                features[f'zone_{zone}_pct'] = zones.count(zone) / len(v13_rally)
            
            # Angular velocity from v13
            angular = [f.get('angular_vel', {}) for f in v13_rally if f.get('angular_vel')]
            if angular:
                for key in ['l_forearm', 'r_forearm', 'torso']:
                    vals = [a.get(key, 0) for a in angular if a.get(key)]
                    if vals:
                        features[f'v13_angvel_{key}_mean'] = np.mean(vals)
        
        # Rally duration
        features['rally_duration'] = end - start + 1
        
        X_list.append(features)
        y_list.append(int(row['winner']))
    
    return pd.DataFrame(X_list).fillna(0), np.array(y_list)

X, y = build_combined_features(labels, v11_df, v13_frames, v11_feature_cols)
print(f"    Combined features: {X.shape[1]} columns, {len(X)} samples")

# Feature selection + Cross-validation
print("\n[5] Cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7777)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []

# Test with variance threshold selection
for threshold in [0.0, 0.1, 0.2]:
    if threshold == 0.0:
        X_sel = X_scaled
    else:
        selector = VarianceThreshold(threshold=threshold)
        X_sel = selector.fit_transform(X_scaled)
    
    n_feat = X_sel.shape[1]
    
    # RF
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=7777)
    scores = cross_val_score(clf, X_sel, y, cv=cv, scoring='accuracy')
    results.append(('RF', f'thresh={threshold}', n_feat, scores.mean(), scores.std()))
    print(f"    RF thresh={threshold}, features={n_feat}: CV={scores.mean():.3f} +/- {scores.std():.3f}")

# Without selection - test different models
clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=7777)
scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
results.append(('RF', 'full', X_scaled.shape[1], scores.mean(), scores.std()))
print(f"    RF full: CV={scores.mean():.3f} +/- {scores.std():.3f}")

clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=7777)
scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
results.append(('GB', 'full', X_scaled.shape[1], scores.mean(), scores.std()))
print(f"    GB full: CV={scores.mean():.3f} +/- {scores.std():.3f}")

# Find best
best = max(results, key=lambda x: x[3])
print(f"\n    Best: {best[0]} {best[1]} -> CV={best[3]:.3f} ({best[2]} features)")

# Save best model
print("\n[6] Saving model...")
clf_final = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=7777)
clf_final.fit(X_scaled, y)

with open(f"{ROOT}/models/v44.pkl", "wb") as f:
    pickle.dump((clf_final, scaler), f)

# Save features info
feature_info = {
    'n_features': X.shape[1],
    'feature_names': list(X.columns),
    'best_cv': best[3],
    'best_model': f"{best[0]} {best[1]}",
    'combined': ['v11 (angular velocity)', 'v13 (landing/momentum)']
}
with open(f"{ROOT}/models/v44_features.json", "w") as f:
    json.dump(feature_info, f, indent=2)

print(f"\n    Saved: models/v44.pkl")
print(f"    Features: {X.shape[1]} columns")
print(f"    Best CV: {best[3]:.3f}")
print(f"    v41 best: 0.769")
print(f"    Delta: {best[3] - 0.769:+.3f}")