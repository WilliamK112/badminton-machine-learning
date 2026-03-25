#!/usr/bin/env python3
"""
train_v43.py - Train with v13 features (landing prediction, momentum, court zone)
Goal: Beat v41 CV=0.769
"""
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

ROOT = "/Users/William/.openclaw/workspace/projects/badminton-ai"

print("=" * 60)
print("TRAIN v43 - v13 Features (Landing Prediction + Momentum)")
print("=" * 60)

# Load v13 frame features
print("\n[1] Loading v13 frame features...")
v13_frames = []
with open(f"{ROOT}/data/frame_features_v13.jsonl", "r") as f:
    for line in f:
        v13_frames.append(json.loads(line))
print(f"    Loaded {len(v13_frames)} frames")

# Load v9 labels (balanced 28:14)
print("[2] Loading labels...")
labels = pd.read_csv(f"{ROOT}/data/rally_labels_v9.csv.gz")
print(f"    Labels: {len(labels)} rallies (28 winner A, 14 winner B)")

# Build rally-level dataset from v13 features
print("\n[3] Building rally-level features...")

def encode_zone(zone_str):
    """Encode court zone to numeric"""
    zone_map = {
        'front_X': 0, 'mid_X': 1, 'net': 2,
        'front_Y': 3, 'mid_Y': 4, 'back_Y': 5
    }
    return zone_map.get(zone_str, -1)

def build_v13_features(labels_df, v13_frames):
    """Extract rally-level features from v13 frame data"""
    # Create frame lookup
    frame_lookup = {f['frame']: f for f in v13_frames}
    
    X_list = []
    y_list = []
    
    for _, row in labels_df.iterrows():
        start = int(row['start_frame'])
        end = int(row['end_frame'])
        
        # Get frames in this rally
        rally_frames = []
        for i in range(start, end+1):
            f = frame_lookup.get(i)
            if f is not None and f.get('players') is not None:
                rally_frames.append(f)
        
        if len(rally_frames) == 0:
            print(f"    Warning: No valid frames for rally {start}-{end}")
            continue
        
        features = {}
        
        # === Player position features ===
        for player in ['X', 'Y']:
            centers = []
            for f in rally_frames:
                try:
                    if f.get('players') and f['players'].get(player) and f['players'][player].get('center'):
                        centers.append(f['players'][player]['center'])
                except:
                    pass
            if centers:
                cx = [c[0] for c in centers]
                cy = [c[1] for c in centers]
                features[f'{player}_center_x_mean'] = np.mean(cx)
                features[f'{player}_center_x_std'] = np.std(cx)
                features[f'{player}_center_y_mean'] = np.mean(cy)
                features[f'{player}_center_y_std'] = np.std(cy)
                # Movement speed
                if len(cx) > 1:
                    vx = np.diff(cx)
                    vy = np.diff(cy)
                    speed = np.sqrt(np.array(vx)**2 + np.array(vy)**2)
                    features[f'{player}_speed_mean'] = np.mean(speed)
                    features[f'{player}_speed_max'] = np.max(speed)
                # Stance width
                sw = [f.get(f'{player}_stance_width', 0) for f in rally_frames if f.get(f'{player}_stance_width')]
                if sw:
                    features[f'{player}_stance_mean'] = np.mean(sw)
                    features[f'{player}_stance_std'] = np.std(sw)
        
        # === Shuttle features ===
        # Shuttle visibility
        visible = [f['shuttle']['visible'] for f in rally_frames if 'shuttle' in f]
        if visible:
            features['shuttle_visible_pct'] = sum(visible) / len(visible)
        
        # Shuttle positions
        shuttle_xy = [f['shuttle']['xy'] for f in rally_frames if f.get('shuttle', {}).get('xy')]
        if shuttle_xy:
            sx = [s[0] for s in shuttle_xy]
            sy = [s[1] for s in shuttle_xy]
            features['shuttle_x_mean'] = np.mean(sx)
            features['shuttle_x_std'] = np.std(sx)
            features['shuttle_y_mean'] = np.mean(sy)
            features['shuttle_y_std'] = np.std(sy)
            # Shuttle speed
            if len(sx) > 1:
                svx = np.diff(sx)
                svy = np.diff(sy)
                s_speed = np.sqrt(np.array(svx)**2 + np.array(svy)**2)
                features['shuttle_speed_mean'] = np.mean(s_speed)
                features['shuttle_speed_max'] = np.max(s_speed)
        
        # Shuttle direction change
        dir_changes = [f.get('shuttle_dir_change', 0) for f in rally_frames if f.get('shuttle_dir_change') is not None]
        if dir_changes:
            features['shuttle_dir_change_mean'] = np.mean(dir_changes)
            features['shuttle_dir_change_max'] = np.max(dir_changes)
            features['shuttle_dir_change_sum'] = np.sum(dir_changes)
        
        # === v13 NEW FEATURES ===
        # Predicted landing positions
        pred_land_x = [f.get('predicted_landing_x', -1) for f in rally_frames if f.get('predicted_landing_x', -1) > 0]
        pred_land_y = [f.get('predicted_landing_y', -1) for f in rally_frames if f.get('predicted_landing_y', -1) > 0]
        if pred_land_x:
            features['pred_landing_x_mean'] = np.mean(pred_land_x)
            features['pred_landing_x_std'] = np.std(pred_land_x)
            features['pred_landing_y_mean'] = np.mean(pred_land_y)
            features['pred_landing_y_std'] = np.std(pred_land_y)
            features['pred_landing_count'] = len(pred_land_x) / len(rally_frames)
        
        # Shuttle momentum
        momentum = [f.get('shuttle_momentum', 0) for f in rally_frames if f.get('shuttle_momentum') is not None]
        if momentum:
            features['momentum_mean'] = np.mean(momentum)
            features['momentum_std'] = np.std(momentum)
            features['momentum_max'] = np.max(np.abs(momentum))
        
        # Court zone distribution
        zones = [f.get('court_zone', 'net') for f in rally_frames]
        zone_counts = {}
        for z in zones:
            zone_counts[z] = zone_counts.get(z, 0) + 1
        for zone in ['front_X', 'mid_X', 'net', 'front_Y', 'mid_Y', 'back_Y']:
            features[f'zone_{zone}_pct'] = zone_counts.get(zone, 0) / len(rally_frames)
        
        # Angular velocity (if available)
        angular = [f.get('angular_vel', {}) for f in rally_frames if f.get('angular_vel')]
        if angular:
            for key in ['l_forearm', 'r_forearm', 'torso']:
                vals = [a.get(key, 0) for a in angular if a.get(key)]
                if vals:
                    features[f'angvel_{key}_mean'] = np.mean(vals)
                    features[f'angvel_{key}_max'] = np.max(vals)
        
        # Rally duration
        features['rally_duration_frames'] = len(rally_frames)
        
        X_list.append(features)
        y_list.append(int(row['winner']))
    
    return pd.DataFrame(X_list).fillna(0), np.array(y_list)

X, y = build_v13_features(labels, v13_frames)
print(f"    Features: {X.shape[1]} columns")
print(f"    Samples: {len(X)} rallies")

# Cross-validation
print("\n[4] Cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7777)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []

# Test RF with different depths
for depth in [3, 5, 7]:
    clf = RandomForestClassifier(n_estimators=100, max_depth=depth, class_weight='balanced', random_state=7777)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    results.append(('RF', f'depth={depth}', scores.mean(), scores.std()))
    print(f"    RF depth={depth}: CV={scores.mean():.3f} +/- {scores.std():.3f}")

# GradientBoosting
for depth in [2, 3, 4]:
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=depth, random_state=7777)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    results.append(('GB', f'depth={depth}', scores.mean(), scores.std()))
    print(f"    GB depth={depth}: CV={scores.mean():.3f} +/- {scores.std():.3f}")

# Logistic Regression
for C in [0.1, 1.0, 10.0]:
    clf = LogisticRegression(C=C, max_iter=1000, class_weight='balanced', random_state=7777)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    results.append(('LR', f'C={C}', scores.mean(), scores.std()))
    print(f"    LR C={C}: CV={scores.mean():.3f} +/- {scores.std():.3f}")

# Find best
best = max(results, key=lambda x: x[2])
print(f"\n    Best: {best[0]} {best[1]} -> CV={best[2]:.3f}")

# Save best model
print("\n[5] Saving model...")
clf_final = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=7777)
clf_final.fit(X_scaled, y)

with open(f"{ROOT}/models/v43.pkl", "wb") as f:
    pickle.dump((clf_final, scaler), f)

# Save features info
feature_info = {
    'n_features': X.shape[1],
    'feature_names': list(X.columns),
    'best_cv': best[2],
    'best_model': f"{best[0]} {best[1]}",
    'v13_new_features': ['predicted_landing_x/y', 'court_zone', 'shuttle_momentum', 'angular_vel']
}
with open(f"{ROOT}/models/v43_features.json", "w") as f:
    json.dump(feature_info, f, indent=2)

print(f"\n    Saved: models/v43.pkl")
print(f"    Features: {X.shape[1]} columns")
print(f"    Best CV: {best[2]:.3f}")
print(f"    v41 best: 0.769")
print(f"    Delta: {best[2] - 0.769:+.3f}")