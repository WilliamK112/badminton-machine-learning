"""
Feature Selection + Model Training v48
Uses SelectKBest and RF importance to remove noisy motion features
Goal: Improve on v46 (CV=0.889) by removing noisy features
"""
import pandas as pd
import numpy as np
import json
import gzip
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load frame features v6
def load_frame_features():
    frames = []
    with gzip.open('data/frame_features_v6.jsonl.gz', 'rt') as f:
        for line in f:
            frames.append(json.loads(line))
    df = pd.DataFrame(frames)
    return df

# Load body features v14
def load_body_features():
    df = pd.read_csv('data/body_features_v14.csv')
    return df

# Load motion quantification v1
def load_motion_features():
    df = pd.read_csv('data/motion_quant_v1.csv')
    return df

print("Loading features...")
frame_df = load_frame_features()
body_df = load_body_features()
motion_df = load_motion_features()

print(f"Frame features: {frame_df.shape}")
print(f"Body features: {body_df.shape}")
print(f"Motion features: {motion_df.shape}")

# Align by frame - all use 'frame' column
merged = frame_df.merge(body_df, on='frame', how='inner')
merged = merged.merge(motion_df, on='frame', how='inner')
print(f"Merged: {merged.shape}")

# Load labels - use winner_proxy from quant_features_v5
labels_df = pd.read_csv('data/quant_features_v5.csv')[['frame', 'winner_proxy']]
labels_df = labels_df.fillna(0)

# Create target: 1 if P1 wins point, 0 otherwise (winner_proxy = 1 means P1)
merged = merged.merge(labels_df[['frame', 'winner_proxy']], on='frame', how='inner')

# Filter to only labeled frames
merged = merged[merged['winner_proxy'].isin([0, 1])].copy()
print(f"With labels: {merged.shape}")

# Only use numeric columns as features
numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['frame', 'winner_proxy', 'p1_point', 'p2_point', 'p1_rally', 'p2_rally']
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

print(f"Numeric feature columns: {len(feature_cols)}")

X = merged[feature_cols].values
y = merged['winner_proxy'].values

# Handle any NaN values
X = np.nan_to_num(X, nan=0.0)

print(f"Total features: {len(feature_cols)}")
print(f"Samples: {len(X)}")

# Feature Selection using SelectKBest
print("\n=== Feature Selection ===")
selector = SelectKBest(f_classif, k='all')
selector.fit(X, y)
scores = selector.scores_

# Rank features by score
feature_scores = list(zip(feature_cols, scores))
feature_scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop 20 features by F-score:")
for name, score in feature_scores[:20]:
    print(f"  {name}: {score:.3f}")

# Try different k values
print("\n=== Model Training with Feature Selection ===")
best_cv = 0
best_k = 0

for k in [10, 15, 20, 25, 30, 35, 40]:
    # Select top k features
    selector_k = SelectKBest(f_classif, k=k)
    X_selected = selector_k.fit_transform(X, y)
    
    # Train RF classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
    cv_scores = cross_val_score(rf, X_selected, y, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    
    print(f"k={k}: CV accuracy = {cv_mean:.3f} (+/- {cv_scores.std():.3f})")
    
    if cv_mean > best_cv:
        best_cv = cv_mean
        best_k = k

print(f"\nBest k={best_k} with CV={best_cv:.3f}")

# Also try RF importance-based selection
print("\n=== RF Importance Selection ===")
rf_full = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
rf_full.fit(X, y)
importances = rf_full.feature_importances_

feature_importance = list(zip(feature_cols, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("\nTop 20 features by RF importance:")
for name, imp in feature_importance[:20]:
    print(f"  {name}: {imp:.4f}")

# Select top k by importance
top_features = [f[0] for f in feature_importance[:best_k]]
X_importance = merged[top_features].values
X_importance = np.nan_to_num(X_importance, nan=0.0)

rf_imp = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
cv_scores_imp = cross_val_score(rf_imp, X_importance, y, cv=5, scoring='accuracy')
print(f"Top {best_k} by importance: CV = {cv_scores_imp.mean():.3f}")

# Train final model with best features
print("\n=== Final Model ===")
final_features = [f[0] for f in feature_importance[:best_k]]
X_final = merged[final_features].values
X_final = np.nan_to_num(X_final, nan=0.0)

rf_final = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
cv_final = cross_val_score(rf_final, X_final, y, cv=5, scoring='accuracy')
print(f"Final model (k={best_k}): CV = {cv_final.mean():.3f}")

# Compare with v46
print(f"\n=== Comparison ===")
print(f"v46 (35 features, body+v6): CV = 0.889")
print(f"v47 (67 features, all): CV = 0.815")
print(f"v48 (feature selection): CV = {cv_final.mean():.3f}")

# Save results
results = {
    'version': 'v48',
    'best_k': best_k,
    'cv_score': float(cv_final.mean()),
    'top_features': final_features[:20],
    'total_features': len(feature_cols)
}

with open('model_results_v48.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to model_results_v48.json")