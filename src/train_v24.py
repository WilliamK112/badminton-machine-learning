#!/usr/bin/env python3
"""Train v24: Add temporal/rally dynamics features to beat v19 (CV 0.733)"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Load rally labels
with gzip.open(ROOT / 'data/rally_labels_v9.csv.gz', 'rt') as f:
    rally_v9 = pd.read_csv(f)
print(f"Loaded {len(rally_v9)} rallies")

# Load quant features (has angular velocity)
with gzip.open(ROOT / 'data/quant_features_v11.csv.gz', 'rt') as f:
    quant = pd.read_csv(f)
print(f"Loaded {len(quant)} frame features with angular velocity")

# Load frame features for stance
import json as json_lib
frame_features = []
with open(ROOT / 'data/frame_features_v12.jsonl', 'r') as f:
    for line in f:
        frame_features.append(json_lib.loads(line))
frame_df = pd.DataFrame(frame_features)
print(f"Loaded {len(frame_df)} frame features v12")

# Build rally-level features with more dynamics
features_list = []
for _, row in rally_v9.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    rally_frames = quant[(quant['frame'] >= start) & (quant['frame'] <= end)]
    rally_frames_frame = frame_df[(frame_df['frame'] >= start) & (frame_df['frame'] <= end)]
    
    if len(rally_frames) < 3:
        continue
    
    # Basic features
    end_row = rally_frames.iloc[-1]
    shuttle_x_end = end_row['shuttle_x']
    shuttle_y_end = end_row['shuttle_y']
    dist_from_center = abs(shuttle_x_end - 0.5)
    is_deep = 1 if shuttle_y_end > 0.7 else 0
    rally_duration = end - start
    
    feat_dict = {
        'rally_idx': len(features_list),
        'winner': int(row['winner']),
        'shuttle_x_end': shuttle_x_end,
        'shuttle_y_end': shuttle_y_end,
        'dist_from_center': dist_from_center,
        'is_deep': is_deep,
        'rally_duration': rally_duration,
    }
    
    # Angular velocity features (from quant)
    for player in ['X', 'Y']:
        for joint in ['arms', 'torso', 'legs']:
            col = f"{player}_{joint}_ang_vel"
            if col in rally_frames.columns:
                vals = rally_frames[col].dropna()
                if len(vals) > 0:
                    feat_dict[f'{player}_{joint}_ang_vel_max'] = vals.max()
                    feat_dict[f'{player}_{joint}_ang_vel_mean'] = vals.mean()
                    feat_dict[f'{player}_{joint}_ang_vel_std'] = vals.std() if len(vals) > 1 else 0
    
    # Motion features
    feat_dict['avg_motion'] = row.get('avg_motion', 0)
    feat_dict['max_motion'] = row.get('max_motion', 0)
    
    # Shuttle trajectory dynamics
    if 'shuttle_speed' in rally_frames.columns:
        speed_vals = rally_frames['shuttle_speed'].dropna()
        if len(speed_vals) > 0:
            feat_dict['shuttle_speed_max'] = speed_vals.max()
            feat_dict['shuttle_speed_mean'] = speed_vals.mean()
            feat_dict['shuttle_speed_change'] = speed_vals.iloc[-1] - speed_vals.iloc[0] if len(speed_vals) > 1 else 0
    
    # Position changes (momentum)
    for player in ['X', 'Y']:
        if f'{player}_center_x' in rally_frames.columns:
            x_vals = rally_frames[f'{player}_center_x'].dropna()
            y_vals = rally_frames[f'{player}_center_y'].dropna()
            if len(x_vals) > 1:
                feat_dict[f'{player}_x_range'] = x_vals.max() - x_vals.min()
                feat_dict[f'{player}_y_range'] = y_vals.max() - y_vals.min()
                feat_dict[f'{player}_total_dist'] = np.sum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
    
    # Stance features
    for player in ['X', 'Y']:
        stance_col = f'{player}_stance_width'
        if stance_col in rally_frames_frame.columns:
            stance_vals = pd.to_numeric(rally_frames_frame[stance_col], errors='coerce').dropna()
            if len(stance_vals) > 0:
                feat_dict[f'{player}_stance_mean'] = stance_vals.mean()
                feat_dict[f'{player}_stance_std'] = stance_vals.std() if len(stance_vals) > 1 else 0
    
    features_list.append(feat_dict)

df = pd.DataFrame(features_list)
print(f"Built {len(df)} rally samples with {len(df.columns)-2} features")

feature_cols = [c for c in df.columns if c not in ['rally_idx', 'winner']]
X = df[feature_cols].values
y = df['winner'].values

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

selector = SelectKBest(f_classif, k=min(12, len(feature_cols)))
X_selected = selector.fit_transform(X, y)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected features ({len(selected_features)}): {selected_features}")

print(f"Class distribution: winner=1: {sum(y==1)}, winner=0: {sum(y==0)}")

# Try multiple models
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'RF': RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, random_state=42, class_weight='balanced'),
    'GB': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    'LR': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
}

best_score = 0
best_model_name = None
results = []

for name, model in models.items():
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale for LR
        if name == 'LR':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        m = model.__class__(**model.get_params())
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        cv_scores.append(balanced_accuracy_score(y_test, pred))
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    results.append({'model': name, 'score': mean_score, 'std': std_score})
    print(f"{name}: CV {mean_score:.3f} (+/- {std_score:.3f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name

print(f"\nBest: {best_model_name} with CV {best_score:.3f}")

# Temporal test
test_size = 8
X_train, X_test = X_selected[:-test_size], X_selected[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

if best_model_name == 'LR':
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

model_test = models[best_model_name].__class__(**models[best_model_name].get_params())
model_test.fit(X_train, y_train)
y_pred = model_test.predict(X_test)

temporal_acc = accuracy_score(y_test, y_pred)
temporal_bal = balanced_accuracy_score(y_test, y_pred)

print(f"Temporal test: acc={temporal_acc:.3f}, balanced={temporal_bal:.3f}")

# Save report
report = {
    'timestamp_local': datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-05:00',
    'step_executed': 'Train v24 - More dynamics features',
    'samples': len(df),
    'features': len(selected_features),
    'feature_names': selected_features,
    'class_distribution': {'winner_1': int(sum(y==1)), 'winner_0': int(sum(y==0))},
    'best_model': best_model_name,
    'cv_balanced_accuracy': round(best_score, 3),
    'all_results': results,
    'temporal_test': {
        'accuracy': round(temporal_acc, 3),
        'balanced_accuracy': round(temporal_bal, 3),
    },
    'vs_v19': f"v19 CV was 0.733, v24 {'improved' if best_score > 0.733 else 'no improvement'}"
}

out_path = ROOT / 'reports' / f"train_v24_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"Saved: {out_path}")