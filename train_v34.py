#!/usr/bin/env python3
"""
train_v34.py - Try balanced labels (v9) with simpler model
Heartbeat step: Use v9 labels (28:14 balanced) with simpler features - might generalize better with balanced data
"""
import gzip
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ROOT = "/Users/William/.openclaw/workspace/projects/badminton-ai"

# Load v5 features (simpler, more coverage)
print("Loading v5 features...")
with gzip.open(f"{ROOT}/data/quant_features_v5.csv.gz", 'rt') as f:
    df = pd.read_csv(f)

# Try v9 labels (balanced) - check which features they can use
print("Loading v9 labels...")
df_labels = pd.read_csv(f"{ROOT}/data/rally_labels_v9.csv.gz")

print(f"v5 features: {len(df)} frames, columns: {list(df.columns)[:10]}")
print(f"v9 labels: {len(df_labels)} rallies")
print(f"Class distribution: {dict(df_labels['winner'].value_counts())}")

# Check what's available in v5
print("\nAvailable columns in v5:", df.columns.tolist())

# Build simple features that work with v5
def build_simple_features(df, df_labels):
    features = []
    for idx, row in df_labels.iterrows():
        sf, ef = int(row['start_frame']), int(row['end_frame'])
        if sf >= len(df) or ef >= len(df):
            continue
        rally_df = df[(df['frame'] >= sf) & (df['frame'] <= ef)]
        if len(rally_df) == 0:
            continue
        
        # Get available columns
        feat = {'winner': row['winner']}
        
        # Basic shuttle position if available
        if 'shuttle_y' in rally_df.columns:
            feat['shuttle_y_end'] = rally_df['shuttle_y'].iloc[-1]
            feat['shuttle_y_start'] = rally_df['shuttle_y'].iloc[0]
            feat['shuttle_y_range'] = rally_df['shuttle_y'].max() - rally_df['shuttle_y'].min()
        
        if 'shuttle_x' in rally_df.columns:
            feat['shuttle_x_end'] = rally_df['shuttle_x'].iloc[-1]
            feat['shuttle_x_range'] = rally_df['shuttle_x'].max() - rally_df['shuttle_x'].min()
        
        # Motion features if available
        if 'shuttle_speed' in rally_df.columns:
            feat['avg_speed'] = rally_df['shuttle_speed'].mean()
            feat['max_speed'] = rally_df['shuttle_speed'].max()
            feat['min_speed'] = rally_df['shuttle_speed'].min()
        
        # Player position if available
        if 'player_y' in rally_df.columns:
            feat['player_y_end'] = rally_df['player_y'].iloc[-1]
            feat['player_y_range'] = rally_df['player_y'].max() - rally_df['player_y'].min()
        
        # Rally duration
        feat['duration'] = ef - sf
        feat['frame_count'] = len(rally_df)
        
        features.append(feat)
    return pd.DataFrame(features)

df_feat = build_simple_features(df, df_labels)
print(f"\nRallies with features: {len(df_feat)}")
print(f"Class distribution: {dict(df_feat['winner'].value_counts())}")
print(f"Feature columns: {df_feat.columns.tolist()}")

# Use all numeric columns except 'winner' as features
feature_cols = [c for c in df_feat.columns if c != 'winner']
X = df_feat[feature_cols].fillna(0).values
y = df_feat['winner'].values

print(f"\nFeature matrix shape: {X.shape}")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validate with multiple models
print("\n=== Cross-Validation ===")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Use 3-fold due to small sample size

models = {
    'RF_balanced': RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42),
    'RF': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
    'LR_balanced': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'SVM': SVC(class_weight='balanced', random_state=42),
}

results = {}
for name, model in models.items():
    try:
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='balanced_accuracy')
        results[name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
        print(f"{name}: CV={scores.mean():.3f}+/-{scores.std():.3f}")
    except Exception as e:
        print(f"{name}: Error - {e}")

best_model = max(results.items(), key=lambda x: x[1]['mean'])
print(f"\nBest: {best_model[0]} with CV={best_model[1]['mean']:.3f}")

# Save report
report = {
    "timestamp_local": "2026-03-24T11-12-00-05:00",
    "step_executed": "Try v9 balanced labels with simpler features (v5)",
    "samples": len(df_feat),
    "class_distribution": {"winner_0": int((y==0).sum()), "winner_1": int((y==1).sum())},
    "best_model": best_model[0],
    "best_cv": float(best_model[1]['mean']),
    "all_results": {k: {'mean': v['mean'], 'std': v['std']} for k, v in results.items()},
    "baseline_v19_cv": 0.733,
    "v31_cv": 0.900,
}

with open(f"{ROOT}/reports/train_v34_2026-03-24T11-12-00.json", 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n=== Summary ===")
print(f"v9 labels (balanced): {len(df_feat)} samples, {int((y==0).sum())}:{int((y==1).sum())}")
print(f"Best model: {best_model[0]} with CV={best_model[1]['mean']:.3f}")
print(f"Previous v31 (v4+class_weight): CV=0.900")
print("Report saved.")